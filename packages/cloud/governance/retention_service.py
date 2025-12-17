# -*- coding: utf-8 -*-
"""
GDPR Phase 4: Data Retention Service with Legal Hold Support.

This module implements comprehensive data retention management per GDPR Art. 5(1)(e):
- Retention policy registry per tenant
- Auto-purge scheduler with auditable events
- Legal hold management with strict access control
- Full audit trail for compliance

Design Doc Reference:
    - Phase 4: "Retention per tenant + auto-purge + legal hold"
    - GDPR Article 5(1)(e): Storage Limitation

CLOUD ZONE ONLY.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, Final, List, Optional, Set, Tuple
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Minimum retention periods (days) - CANNOT be reduced below these
MIN_RETENTION_DAYS: Final[Dict[str, int]] = {
    # Compliance data (7 years minimum per financial regulations)
    "approval_records": 2555,  # 7 years
    "access_audits": 2555,
    "break_glass_requests": 2555,
    "dsar_requests": 2555,
    "governance_audit_logs": 2555,
    "legal_hold_records": 2555,
    "billing_records": 2555,
    # Operational data (shorter minimums)
    "telemetry_raw_order_events": 1,
    "telemetry_detailed_non_sensitive": 7,
    "telemetry_aggregated": 30,
    "alerts": 30,
    "commands": 30,
    "config_blobs": 30,
    "deployment_data": 30,
    "run_data": 30,
    "application_logs": 7,
    "distributed_traces": 1,
    "session_data": 1,
}

# Default retention periods (days)
DEFAULT_RETENTION_DAYS: Final[Dict[str, int]] = {
    # Compliance data
    "approval_records": 2555,
    "access_audits": 2555,
    "break_glass_requests": 2555,
    "dsar_requests": 2555,
    "governance_audit_logs": 2555,
    "legal_hold_records": 2555,
    "billing_records": 2555,
    # Telemetry by level
    "telemetry_raw_order_events": 7,
    "telemetry_detailed_non_sensitive": 30,
    "telemetry_aggregated": 90,
    # Operational data
    "alerts": 365,
    "commands": 180,
    "config_blobs": 365,
    "deployment_data": 365,
    "run_data": 365,
    "application_logs": 30,
    "distributed_traces": 7,
    "session_data": 1,
    # User data
    "user_identity": 90,  # Post-deletion retention
    "user_settings": 90,
}

# Maximum retention periods (days) - None means indefinite
MAX_RETENTION_DAYS: Final[Dict[str, Optional[int]]] = {
    "telemetry_raw_order_events": 30,
    "telemetry_detailed_non_sensitive": 90,
    "telemetry_aggregated": 365,
    "session_data": 7,
    "application_logs": 90,
    "distributed_traces": 30,
    # Compliance data can be extended
    "approval_records": 3650,  # 10 years max
    "access_audits": 3650,
    "break_glass_requests": 3650,
    "dsar_requests": 3650,
    # Customer data: indefinite
    "strategy_versions": None,
    "backtest_results": None,
    "config_blobs": None,
}

# Data categories that require 7-year minimum retention
COMPLIANCE_DATA_CATEGORIES: Final[Set[str]] = {
    "approval_records",
    "access_audits",
    "break_glass_requests",
    "dsar_requests",
    "governance_audit_logs",
    "legal_hold_records",
    "billing_records",
}

# All valid data categories
ALL_DATA_CATEGORIES: Final[Set[str]] = {
    "telemetry_raw_order_events",
    "telemetry_detailed_non_sensitive",
    "telemetry_aggregated",
    "alerts",
    "commands",
    "config_blobs",
    "deployment_data",
    "run_data",
    "approval_records",
    "access_audits",
    "break_glass_requests",
    "dsar_requests",
    "governance_audit_logs",
    "legal_hold_records",
    "billing_records",
    "application_logs",
    "distributed_traces",
    "session_data",
    "user_identity",
    "user_settings",
    "strategy_versions",
    "backtest_results",
    "research_artifacts",
}


class RetentionAction(str, Enum):
    """Actions for data after retention period expires."""
    DELETE = "delete"
    ARCHIVE = "archive"
    ANONYMIZE = "anonymize"
    AGGREGATE = "aggregate"


class PurgeStatus(str, Enum):
    """Status of a purge operation."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    PARTIAL = "partial"


class LegalHoldStatus(str, Enum):
    """Status of a legal hold."""
    ACTIVE = "active"
    RELEASED = "released"
    EXPIRED = "expired"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RetentionPolicy:
    """
    Retention policy for a data type within a workspace.

    Attributes:
        id: Unique identifier
        workspace_id: Workspace this policy belongs to
        data_type: Type of data this policy applies to
        retention_days: Number of days to retain data
        action: What to do after retention period
        auto_purge_enabled: Whether to automatically purge
        last_purge_at: When the last purge was executed
        created_at: When policy was created
        updated_at: When policy was last updated
        created_by: User who created the policy
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    data_type: str = ""
    retention_days: int = 90
    action: RetentionAction = RetentionAction.DELETE
    auto_purge_enabled: bool = True
    last_purge_at: Optional[datetime] = None
    last_purge_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate and adjust retention days to comply with minimums."""
        min_days = MIN_RETENTION_DAYS.get(self.data_type, 1)
        if self.retention_days < min_days:
            self.retention_days = min_days

        max_days = MAX_RETENTION_DAYS.get(self.data_type)
        if max_days is not None and self.retention_days > max_days:
            self.retention_days = max_days

    @property
    def cutoff_date(self) -> datetime:
        """Calculate the cutoff date for this policy."""
        return datetime.now(timezone.utc) - timedelta(days=self.retention_days)

    @property
    def is_compliance_data(self) -> bool:
        """Check if this policy is for compliance-critical data."""
        return self.data_type in COMPLIANCE_DATA_CATEGORIES

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "workspace_id": self.workspace_id,
            "data_type": self.data_type,
            "retention_days": self.retention_days,
            "action": self.action.value,
            "auto_purge_enabled": self.auto_purge_enabled,
            "last_purge_at": self.last_purge_at.isoformat() if self.last_purge_at else None,
            "last_purge_count": self.last_purge_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "is_compliance_data": self.is_compliance_data,
            "cutoff_date": self.cutoff_date.isoformat(),
        }


@dataclass
class LegalHold:
    """
    Legal hold on data - prevents purge until released.

    Attributes:
        id: Unique identifier
        workspace_id: Workspace scope
        data_type: Data type under hold
        reason: Legal justification for hold
        hold_until: Optional expiry date (None = indefinite)
        status: Current status
        created_by: User who created the hold
        created_at: When hold was created
        released_by: User who released the hold
        released_at: When hold was released
        release_reason: Reason for releasing
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    data_type: str = ""
    reason: str = ""
    hold_until: Optional[datetime] = None
    status: LegalHoldStatus = LegalHoldStatus.ACTIVE
    created_by: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    released_by: Optional[str] = None
    released_at: Optional[datetime] = None
    release_reason: Optional[str] = None

    @property
    def is_active(self) -> bool:
        """Check if hold is currently active."""
        if self.status != LegalHoldStatus.ACTIVE:
            return False
        if self.hold_until and datetime.now(timezone.utc) > self.hold_until:
            return False
        return True

    @property
    def duration_days(self) -> int:
        """Calculate duration of hold in days."""
        end = self.released_at or datetime.now(timezone.utc)
        return (end - self.created_at).days

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "workspace_id": self.workspace_id,
            "data_type": self.data_type,
            "reason": self.reason,
            "hold_until": self.hold_until.isoformat() if self.hold_until else None,
            "status": self.status.value,
            "is_active": self.is_active,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "released_by": self.released_by,
            "released_at": self.released_at.isoformat() if self.released_at else None,
            "release_reason": self.release_reason,
            "duration_days": self.duration_days,
        }


@dataclass
class PurgeEvent:
    """
    Audit event for a purge operation.

    Attributes:
        event_id: Unique identifier
        event_type: Type of event
        workspace_id: Workspace scope
        data_type: Data type purged
        status: Outcome status
        retention_days: Policy retention days
        cutoff_date: Data older than this was purged
        records_deleted: Count of deleted records
        records_archived: Count of archived records
        records_anonymized: Count of anonymized records
        records_aggregated: Count of aggregated records
        started_at: When purge started
        completed_at: When purge completed
        executor: Who/what executed the purge
        error_message: Error details if failed
        legal_hold_blocked: Whether blocked by legal hold
    """
    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: str = "purge_completed"
    workspace_id: str = ""
    data_type: str = ""
    status: PurgeStatus = PurgeStatus.COMPLETED
    retention_days: int = 0
    cutoff_date: Optional[datetime] = None
    records_deleted: int = 0
    records_archived: int = 0
    records_anonymized: int = 0
    records_aggregated: int = 0
    bytes_freed: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    executor: str = "scheduler"
    error_message: Optional[str] = None
    legal_hold_blocked: bool = False
    skip_reason: Optional[str] = None

    @property
    def total_records_processed(self) -> int:
        """Calculate total records affected."""
        return (
            self.records_deleted +
            self.records_archived +
            self.records_anonymized +
            self.records_aggregated
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "workspace_id": self.workspace_id,
            "data_type": self.data_type,
            "status": self.status.value,
            "retention_config": {
                "retention_days": self.retention_days,
                "cutoff_date": self.cutoff_date.isoformat() if self.cutoff_date else None,
            },
            "results": {
                "records_deleted": self.records_deleted,
                "records_archived": self.records_archived,
                "records_anonymized": self.records_anonymized,
                "records_aggregated": self.records_aggregated,
                "total_processed": self.total_records_processed,
                "bytes_freed": self.bytes_freed,
            },
            "execution": {
                "started_at": self.started_at.isoformat(),
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                "duration_seconds": self.duration_seconds,
                "executor": self.executor,
            },
            "legal_hold_blocked": self.legal_hold_blocked,
            "skip_reason": self.skip_reason,
            "error": {
                "message": self.error_message,
            } if self.error_message else None,
        }

    def compute_hash(self) -> str:
        """Compute integrity hash for this event."""
        content = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"


@dataclass
class RetentionValidationResult:
    """Result of retention policy validation."""
    is_valid: bool
    effective_days: int
    original_days: int
    adjusted: bool
    reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# Retention Policy Registry
# ============================================================================

class RetentionPolicyRegistry:
    """
    Registry for managing retention policies per workspace.

    Provides GDPR-aligned retention policy management with:
    - Minimum retention enforcement for compliance data
    - Maximum retention limits for sensitive data
    - Audit trail for all policy changes
    """

    def __init__(
        self,
        audit_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        """
        Initialize the registry.

        Args:
            audit_callback: Callback for audit events (action, details)
        """
        self._policies: Dict[str, Dict[str, RetentionPolicy]] = {}  # workspace_id -> data_type -> policy
        self._audit_callback = audit_callback
        self._audit_log: List[Dict[str, Any]] = []

    def create_policy(
        self,
        workspace_id: str,
        data_type: str,
        retention_days: Optional[int] = None,
        action: RetentionAction = RetentionAction.DELETE,
        auto_purge_enabled: bool = True,
        created_by: Optional[str] = None,
    ) -> Tuple[RetentionPolicy, RetentionValidationResult]:
        """
        Create a new retention policy.

        Args:
            workspace_id: Workspace scope
            data_type: Data category
            retention_days: Retention period (uses default if not specified)
            action: Action after retention
            auto_purge_enabled: Enable auto-purge
            created_by: User creating the policy

        Returns:
            Tuple of (created policy, validation result)
        """
        if data_type not in ALL_DATA_CATEGORIES:
            raise ValueError(f"Invalid data type: {data_type}")

        # Get default if not specified
        if retention_days is None:
            retention_days = DEFAULT_RETENTION_DAYS.get(data_type, 90)

        # Validate and adjust
        validation = self._validate_retention_days(data_type, retention_days)

        policy = RetentionPolicy(
            workspace_id=workspace_id,
            data_type=data_type,
            retention_days=validation.effective_days,
            action=action,
            auto_purge_enabled=auto_purge_enabled,
            created_by=created_by,
        )

        # Store policy
        if workspace_id not in self._policies:
            self._policies[workspace_id] = {}
        self._policies[workspace_id][data_type] = policy

        # Audit
        self._log_audit("policy_created", {
            "workspace_id": workspace_id,
            "data_type": data_type,
            "retention_days": validation.effective_days,
            "action": action.value,
            "created_by": created_by,
            "adjusted": validation.adjusted,
            "original_days": validation.original_days,
        })

        return policy, validation

    def get_policy(
        self,
        workspace_id: str,
        data_type: str,
    ) -> Optional[RetentionPolicy]:
        """Get policy for a workspace and data type."""
        return self._policies.get(workspace_id, {}).get(data_type)

    def get_workspace_policies(
        self,
        workspace_id: str,
    ) -> List[RetentionPolicy]:
        """Get all policies for a workspace."""
        return list(self._policies.get(workspace_id, {}).values())

    def get_all_policies(self) -> List[RetentionPolicy]:
        """Get all policies across all workspaces."""
        policies = []
        for ws_policies in self._policies.values():
            policies.extend(ws_policies.values())
        return policies

    def update_policy(
        self,
        workspace_id: str,
        data_type: str,
        retention_days: Optional[int] = None,
        action: Optional[RetentionAction] = None,
        auto_purge_enabled: Optional[bool] = None,
        updated_by: Optional[str] = None,
    ) -> Tuple[Optional[RetentionPolicy], Optional[RetentionValidationResult]]:
        """
        Update an existing retention policy.

        Args:
            workspace_id: Workspace scope
            data_type: Data category
            retention_days: New retention period
            action: New action
            auto_purge_enabled: Enable/disable auto-purge
            updated_by: User making the update

        Returns:
            Tuple of (updated policy, validation result) or (None, None) if not found
        """
        policy = self.get_policy(workspace_id, data_type)
        if not policy:
            return None, None

        old_values = policy.to_dict()
        validation = None

        if retention_days is not None:
            validation = self._validate_retention_days(data_type, retention_days)
            policy.retention_days = validation.effective_days

        if action is not None:
            policy.action = action

        if auto_purge_enabled is not None:
            policy.auto_purge_enabled = auto_purge_enabled

        policy.updated_at = datetime.now(timezone.utc)

        # Audit
        self._log_audit("policy_updated", {
            "workspace_id": workspace_id,
            "data_type": data_type,
            "old_values": old_values,
            "new_values": policy.to_dict(),
            "updated_by": updated_by,
        })

        return policy, validation

    def delete_policy(
        self,
        workspace_id: str,
        data_type: str,
        deleted_by: Optional[str] = None,
    ) -> bool:
        """
        Delete a retention policy.

        Note: Deletion is not allowed for compliance data categories.

        Args:
            workspace_id: Workspace scope
            data_type: Data category
            deleted_by: User deleting the policy

        Returns:
            True if deleted, False otherwise
        """
        if data_type in COMPLIANCE_DATA_CATEGORIES:
            logger.warning(
                f"Cannot delete retention policy for compliance data: {data_type}"
            )
            return False

        ws_policies = self._policies.get(workspace_id, {})
        if data_type not in ws_policies:
            return False

        policy = ws_policies.pop(data_type)

        self._log_audit("policy_deleted", {
            "workspace_id": workspace_id,
            "data_type": data_type,
            "deleted_by": deleted_by,
            "deleted_policy": policy.to_dict(),
        })

        return True

    def validate_retention(
        self,
        data_type: str,
        retention_days: int,
    ) -> RetentionValidationResult:
        """
        Validate a proposed retention period.

        Args:
            data_type: Data category
            retention_days: Proposed retention days

        Returns:
            Validation result with effective days
        """
        return self._validate_retention_days(data_type, retention_days)

    def _validate_retention_days(
        self,
        data_type: str,
        retention_days: int,
    ) -> RetentionValidationResult:
        """Internal validation of retention days."""
        warnings = []
        min_days = MIN_RETENTION_DAYS.get(data_type, 1)
        max_days = MAX_RETENTION_DAYS.get(data_type)
        default_days = DEFAULT_RETENTION_DAYS.get(data_type, 90)

        effective = retention_days
        adjusted = False
        reason = None

        # Check minimum
        if retention_days < min_days:
            effective = min_days
            adjusted = True
            reason = f"Increased to minimum {min_days} days for {data_type}"
            if data_type in COMPLIANCE_DATA_CATEGORIES:
                warnings.append(
                    f"Compliance data requires minimum {min_days // 365} year retention"
                )

        # Check maximum
        if max_days is not None and retention_days > max_days:
            effective = max_days
            adjusted = True
            reason = f"Reduced to maximum {max_days} days for {data_type}"
            warnings.append(f"Maximum retention for {data_type} is {max_days} days")

        # Warn if below default
        if effective < default_days and not adjusted:
            warnings.append(
                f"Retention below recommended default of {default_days} days"
            )

        return RetentionValidationResult(
            is_valid=not adjusted,
            effective_days=effective,
            original_days=retention_days,
            adjusted=adjusted,
            reason=reason,
            warnings=warnings,
        )

    def update_last_purge(
        self,
        workspace_id: str,
        data_type: str,
        purge_count: int,
    ) -> None:
        """Update last purge timestamp for a policy."""
        policy = self.get_policy(workspace_id, data_type)
        if policy:
            policy.last_purge_at = datetime.now(timezone.utc)
            policy.last_purge_count = purge_count

    def _log_audit(self, action: str, details: Dict[str, Any]) -> None:
        """Log an audit event."""
        event = {
            "action": action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **details,
        }
        self._audit_log.append(event)

        if self._audit_callback:
            self._audit_callback(action, details)

    def get_audit_log(
        self,
        workspace_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get audit log, optionally filtered by workspace."""
        if workspace_id:
            logs = [
                e for e in self._audit_log
                if e.get("workspace_id") == workspace_id
            ]
        else:
            logs = list(self._audit_log)

        return logs[-limit:]


# ============================================================================
# Legal Hold Service
# ============================================================================

class LegalHoldService:
    """
    Service for managing legal holds on data.

    Legal holds prevent automatic purge until explicitly released.
    All operations are fully audited.
    """

    def __init__(
        self,
        audit_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        """
        Initialize the service.

        Args:
            audit_callback: Callback for audit events (action, details)
        """
        self._holds: Dict[str, Dict[str, LegalHold]] = {}  # workspace_id -> data_type -> hold
        self._hold_history: List[LegalHold] = []
        self._audit_callback = audit_callback
        self._audit_log: List[Dict[str, Any]] = []

    def create_hold(
        self,
        workspace_id: str,
        data_type: str,
        reason: str,
        created_by: str,
        hold_until: Optional[datetime] = None,
    ) -> LegalHold:
        """
        Create a legal hold on data.

        Args:
            workspace_id: Workspace scope
            data_type: Data type to hold
            reason: Legal justification (REQUIRED)
            created_by: User creating the hold
            hold_until: Optional expiry (None = indefinite)

        Returns:
            Created legal hold
        """
        if not reason or len(reason.strip()) < 10:
            raise ValueError("Legal hold reason must be at least 10 characters")

        if data_type not in ALL_DATA_CATEGORIES:
            raise ValueError(f"Invalid data type: {data_type}")

        # Check for existing active hold
        existing = self.get_active_hold(workspace_id, data_type)
        if existing:
            raise ValueError(
                f"Active legal hold already exists for {data_type} in workspace {workspace_id}"
            )

        hold = LegalHold(
            workspace_id=workspace_id,
            data_type=data_type,
            reason=reason.strip(),
            hold_until=hold_until,
            created_by=created_by,
        )

        # Store hold
        if workspace_id not in self._holds:
            self._holds[workspace_id] = {}
        self._holds[workspace_id][data_type] = hold

        # Audit
        self._log_audit("legal_hold_created", {
            "hold_id": hold.id,
            "workspace_id": workspace_id,
            "data_type": data_type,
            "reason": reason,
            "hold_until": hold_until.isoformat() if hold_until else "indefinite",
            "created_by": created_by,
        })

        return hold

    def get_active_hold(
        self,
        workspace_id: str,
        data_type: str,
    ) -> Optional[LegalHold]:
        """Get active legal hold for a workspace and data type."""
        hold = self._holds.get(workspace_id, {}).get(data_type)
        if hold and hold.is_active:
            return hold
        return None

    def get_workspace_holds(
        self,
        workspace_id: str,
        include_inactive: bool = False,
    ) -> List[LegalHold]:
        """Get all legal holds for a workspace."""
        holds = list(self._holds.get(workspace_id, {}).values())
        if not include_inactive:
            holds = [h for h in holds if h.is_active]
        return holds

    def get_all_active_holds(self) -> List[LegalHold]:
        """Get all active legal holds across all workspaces."""
        holds = []
        for ws_holds in self._holds.values():
            for hold in ws_holds.values():
                if hold.is_active:
                    holds.append(hold)
        return holds

    def is_data_held(
        self,
        workspace_id: str,
        data_type: str,
    ) -> bool:
        """Check if data type is under legal hold."""
        return self.get_active_hold(workspace_id, data_type) is not None

    def release_hold(
        self,
        workspace_id: str,
        data_type: str,
        released_by: str,
        release_reason: str,
    ) -> Optional[LegalHold]:
        """
        Release a legal hold.

        Args:
            workspace_id: Workspace scope
            data_type: Data type
            released_by: User releasing the hold
            release_reason: Justification for release (REQUIRED)

        Returns:
            Released hold or None if not found
        """
        if not release_reason or len(release_reason.strip()) < 10:
            raise ValueError("Release reason must be at least 10 characters")

        hold = self.get_active_hold(workspace_id, data_type)
        if not hold:
            return None

        hold.status = LegalHoldStatus.RELEASED
        hold.released_by = released_by
        hold.released_at = datetime.now(timezone.utc)
        hold.release_reason = release_reason.strip()

        # Move to history
        self._hold_history.append(hold)

        # Audit
        self._log_audit("legal_hold_released", {
            "hold_id": hold.id,
            "workspace_id": workspace_id,
            "data_type": data_type,
            "released_by": released_by,
            "release_reason": release_reason,
            "hold_duration_days": hold.duration_days,
            "original_reason": hold.reason,
        })

        return hold

    def check_and_expire_holds(self) -> List[LegalHold]:
        """
        Check for and expire any holds past their hold_until date.

        Returns:
            List of expired holds
        """
        expired = []
        now = datetime.now(timezone.utc)

        for ws_holds in self._holds.values():
            for hold in ws_holds.values():
                if (
                    hold.status == LegalHoldStatus.ACTIVE and
                    hold.hold_until and
                    now > hold.hold_until
                ):
                    hold.status = LegalHoldStatus.EXPIRED
                    hold.released_at = hold.hold_until  # Expired at scheduled time
                    hold.release_reason = "Automatic expiry at scheduled hold_until date"
                    expired.append(hold)
                    self._hold_history.append(hold)

                    self._log_audit("legal_hold_expired", {
                        "hold_id": hold.id,
                        "workspace_id": hold.workspace_id,
                        "data_type": hold.data_type,
                        "hold_until": hold.hold_until.isoformat(),
                        "hold_duration_days": hold.duration_days,
                    })

        return expired

    def extend_hold(
        self,
        workspace_id: str,
        data_type: str,
        new_hold_until: datetime,
        extended_by: str,
        extension_reason: str,
    ) -> Optional[LegalHold]:
        """
        Extend a legal hold's duration.

        Args:
            workspace_id: Workspace scope
            data_type: Data type
            new_hold_until: New expiry date
            extended_by: User extending the hold
            extension_reason: Justification

        Returns:
            Extended hold or None if not found
        """
        hold = self.get_active_hold(workspace_id, data_type)
        if not hold:
            return None

        old_hold_until = hold.hold_until
        hold.hold_until = new_hold_until

        self._log_audit("legal_hold_extended", {
            "hold_id": hold.id,
            "workspace_id": workspace_id,
            "data_type": data_type,
            "old_hold_until": old_hold_until.isoformat() if old_hold_until else "indefinite",
            "new_hold_until": new_hold_until.isoformat(),
            "extended_by": extended_by,
            "extension_reason": extension_reason,
        })

        return hold

    def get_hold_history(
        self,
        workspace_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[LegalHold]:
        """Get historical legal holds."""
        if workspace_id:
            history = [
                h for h in self._hold_history
                if h.workspace_id == workspace_id
            ]
        else:
            history = list(self._hold_history)

        return history[-limit:]

    def _log_audit(self, action: str, details: Dict[str, Any]) -> None:
        """Log an audit event."""
        event = {
            "action": action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **details,
        }
        self._audit_log.append(event)

        if self._audit_callback:
            self._audit_callback(action, details)

    def get_audit_log(
        self,
        workspace_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get audit log, optionally filtered by workspace."""
        if workspace_id:
            logs = [
                e for e in self._audit_log
                if e.get("workspace_id") == workspace_id
            ]
        else:
            logs = list(self._audit_log)

        return logs[-limit:]


# ============================================================================
# Auto-Purge Scheduler
# ============================================================================

@dataclass
class PurgeSchedulerConfig:
    """Configuration for the auto-purge scheduler."""
    enabled: bool = True
    interval_hours: int = 24
    preferred_hour_utc: int = 3  # 3 AM UTC
    jitter_minutes: int = 30
    batch_size: int = 1000
    max_runtime_minutes: int = 60
    parallel_workspaces: int = 5
    dry_run: bool = False
    notify_before_days: int = 7
    min_records_for_notify: int = 10000


@dataclass
class PurgeSchedulerState:
    """Current state of the purge scheduler."""
    is_running: bool = False
    last_run_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None
    total_runs: int = 0
    total_records_purged: int = 0
    last_run_duration_seconds: float = 0.0
    last_run_workspaces: int = 0
    last_run_errors: int = 0


class AutoPurgeScheduler:
    """
    Automated data purge scheduler.

    Runs periodically to purge data based on retention policies.
    Respects legal holds and produces auditable events.
    """

    def __init__(
        self,
        policy_registry: RetentionPolicyRegistry,
        legal_hold_service: LegalHoldService,
        config: Optional[PurgeSchedulerConfig] = None,
        purge_handler: Optional[
            Callable[[str, str, datetime, RetentionAction, int], int]
        ] = None,
        audit_callback: Optional[Callable[[PurgeEvent], None]] = None,
    ):
        """
        Initialize the scheduler.

        Args:
            policy_registry: Retention policy registry
            legal_hold_service: Legal hold service
            config: Scheduler configuration
            purge_handler: Function to execute purge
                (workspace_id, data_type, cutoff, action, batch_size) -> count
            audit_callback: Callback for purge events
        """
        self.policy_registry = policy_registry
        self.legal_hold_service = legal_hold_service
        self.config = config or PurgeSchedulerConfig()
        self._purge_handler = purge_handler
        self._audit_callback = audit_callback
        self._state = PurgeSchedulerState()
        self._event_log: List[PurgeEvent] = []

    @property
    def state(self) -> PurgeSchedulerState:
        """Get current scheduler state."""
        return self._state

    def run_purge(
        self,
        workspace_id: Optional[str] = None,
        executor: str = "scheduler",
    ) -> List[PurgeEvent]:
        """
        Run purge for all workspaces or a specific workspace.

        Args:
            workspace_id: Optional specific workspace
            executor: Who/what triggered the purge

        Returns:
            List of purge events
        """
        self._state.is_running = True
        self._state.last_run_at = datetime.now(timezone.utc)
        run_start = datetime.now(timezone.utc)

        events: List[PurgeEvent] = []
        workspaces_processed = 0
        errors = 0

        try:
            # Check and expire any legal holds first
            self.legal_hold_service.check_and_expire_holds()

            # Get policies to process
            if workspace_id:
                policies = self.policy_registry.get_workspace_policies(workspace_id)
            else:
                policies = self.policy_registry.get_all_policies()

            for policy in policies:
                if not policy.auto_purge_enabled:
                    continue

                event = self._purge_policy(policy, executor)
                events.append(event)
                self._event_log.append(event)

                if self._audit_callback:
                    self._audit_callback(event)

                if event.status == PurgeStatus.FAILED:
                    errors += 1

            # Count unique workspaces
            workspaces_processed = len(set(p.workspace_id for p in policies))

        finally:
            self._state.is_running = False
            self._state.total_runs += 1
            self._state.last_run_duration_seconds = (
                datetime.now(timezone.utc) - run_start
            ).total_seconds()
            self._state.last_run_workspaces = workspaces_processed
            self._state.last_run_errors = errors
            self._state.total_records_purged += sum(
                e.total_records_processed for e in events
            )
            self._state.next_run_at = (
                datetime.now(timezone.utc) +
                timedelta(hours=self.config.interval_hours)
            )

        return events

    def _purge_policy(
        self,
        policy: RetentionPolicy,
        executor: str,
    ) -> PurgeEvent:
        """Execute purge for a single policy."""
        event = PurgeEvent(
            workspace_id=policy.workspace_id,
            data_type=policy.data_type,
            retention_days=policy.retention_days,
            cutoff_date=policy.cutoff_date,
            executor=executor,
        )

        # Check legal hold
        if self.legal_hold_service.is_data_held(
            policy.workspace_id,
            policy.data_type,
        ):
            event.status = PurgeStatus.SKIPPED
            event.legal_hold_blocked = True
            event.skip_reason = "Data under legal hold"
            event.event_type = "purge_skipped"
            event.completed_at = datetime.now(timezone.utc)
            event.duration_seconds = (
                event.completed_at - event.started_at
            ).total_seconds()

            # Log that purge was blocked
            logger.info(
                f"Purge blocked by legal hold: "
                f"workspace={policy.workspace_id}, data_type={policy.data_type}"
            )
            return event

        # Check if purge handler is configured
        if not self._purge_handler:
            if self.config.dry_run:
                event.status = PurgeStatus.COMPLETED
                event.event_type = "purge_completed_dry_run"
            else:
                event.status = PurgeStatus.SKIPPED
                event.skip_reason = "No purge handler configured"
                event.event_type = "purge_skipped"

            event.completed_at = datetime.now(timezone.utc)
            event.duration_seconds = (
                event.completed_at - event.started_at
            ).total_seconds()
            return event

        try:
            # Execute purge
            if self.config.dry_run:
                # Dry run - don't actually delete
                count = 0
                event.event_type = "purge_completed_dry_run"
            else:
                count = self._purge_handler(
                    policy.workspace_id,
                    policy.data_type,
                    policy.cutoff_date,
                    policy.action,
                    self.config.batch_size,
                )

            # Update counts based on action
            if policy.action == RetentionAction.DELETE:
                event.records_deleted = count
            elif policy.action == RetentionAction.ARCHIVE:
                event.records_archived = count
            elif policy.action == RetentionAction.ANONYMIZE:
                event.records_anonymized = count
            elif policy.action == RetentionAction.AGGREGATE:
                event.records_aggregated = count

            event.status = PurgeStatus.COMPLETED
            if not self.config.dry_run:
                event.event_type = "purge_completed"
            # else: event_type already set to "purge_completed_dry_run"

            # Update policy
            self.policy_registry.update_last_purge(
                policy.workspace_id,
                policy.data_type,
                count,
            )

        except Exception as e:
            logger.error(
                f"Purge failed: workspace={policy.workspace_id}, "
                f"data_type={policy.data_type}, error={e}"
            )
            event.status = PurgeStatus.FAILED
            event.event_type = "purge_failed"
            event.error_message = str(e)

        event.completed_at = datetime.now(timezone.utc)
        event.duration_seconds = (
            event.completed_at - event.started_at
        ).total_seconds()

        return event

    def preview_purge(
        self,
        workspace_id: str,
        data_type: str,
    ) -> Dict[str, Any]:
        """
        Preview what would be purged without executing.

        Args:
            workspace_id: Workspace scope
            data_type: Data type

        Returns:
            Preview information
        """
        policy = self.policy_registry.get_policy(workspace_id, data_type)
        if not policy:
            return {
                "error": "No retention policy found",
                "workspace_id": workspace_id,
                "data_type": data_type,
            }

        hold = self.legal_hold_service.get_active_hold(workspace_id, data_type)

        return {
            "workspace_id": workspace_id,
            "data_type": data_type,
            "policy": policy.to_dict(),
            "cutoff_date": policy.cutoff_date.isoformat(),
            "legal_hold_active": hold is not None,
            "legal_hold": hold.to_dict() if hold else None,
            "would_be_blocked": hold is not None,
            "auto_purge_enabled": policy.auto_purge_enabled,
        }

    def get_event_log(
        self,
        workspace_id: Optional[str] = None,
        status: Optional[PurgeStatus] = None,
        limit: int = 100,
    ) -> List[PurgeEvent]:
        """Get purge event log with optional filters."""
        events = self._event_log

        if workspace_id:
            events = [e for e in events if e.workspace_id == workspace_id]

        if status:
            events = [e for e in events if e.status == status]

        return events[-limit:]

    def get_purge_statistics(
        self,
        workspace_id: Optional[str] = None,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Get purge statistics for the specified period."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        events = [
            e for e in self._event_log
            if e.started_at >= cutoff
        ]

        if workspace_id:
            events = [e for e in events if e.workspace_id == workspace_id]

        successful = [e for e in events if e.status == PurgeStatus.COMPLETED]
        failed = [e for e in events if e.status == PurgeStatus.FAILED]
        skipped = [e for e in events if e.status == PurgeStatus.SKIPPED]
        blocked_by_hold = [e for e in events if e.legal_hold_blocked]

        return {
            "period_days": days,
            "workspace_id": workspace_id,
            "total_runs": len(events),
            "successful_runs": len(successful),
            "failed_runs": len(failed),
            "skipped_runs": len(skipped),
            "blocked_by_legal_hold": len(blocked_by_hold),
            "total_records_deleted": sum(e.records_deleted for e in successful),
            "total_records_archived": sum(e.records_archived for e in successful),
            "total_records_anonymized": sum(e.records_anonymized for e in successful),
            "total_records_aggregated": sum(e.records_aggregated for e in successful),
            "average_duration_seconds": (
                sum(e.duration_seconds for e in successful) / len(successful)
                if successful else 0
            ),
            "by_data_type": self._group_stats_by_data_type(successful),
        }

    def _group_stats_by_data_type(
        self,
        events: List[PurgeEvent],
    ) -> Dict[str, Dict[str, Any]]:
        """Group statistics by data type."""
        by_type: Dict[str, Dict[str, Any]] = {}

        for event in events:
            if event.data_type not in by_type:
                by_type[event.data_type] = {
                    "runs": 0,
                    "records_deleted": 0,
                    "records_archived": 0,
                    "records_anonymized": 0,
                    "records_aggregated": 0,
                }

            by_type[event.data_type]["runs"] += 1
            by_type[event.data_type]["records_deleted"] += event.records_deleted
            by_type[event.data_type]["records_archived"] += event.records_archived
            by_type[event.data_type]["records_anonymized"] += event.records_anonymized
            by_type[event.data_type]["records_aggregated"] += event.records_aggregated

        return by_type


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Constants
    "MIN_RETENTION_DAYS",
    "DEFAULT_RETENTION_DAYS",
    "MAX_RETENTION_DAYS",
    "COMPLIANCE_DATA_CATEGORIES",
    "ALL_DATA_CATEGORIES",
    # Enums
    "RetentionAction",
    "PurgeStatus",
    "LegalHoldStatus",
    # Data classes
    "RetentionPolicy",
    "LegalHold",
    "PurgeEvent",
    "RetentionValidationResult",
    "PurgeSchedulerConfig",
    "PurgeSchedulerState",
    # Services
    "RetentionPolicyRegistry",
    "LegalHoldService",
    "AutoPurgeScheduler",
]
