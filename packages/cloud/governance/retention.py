# -*- coding: utf-8 -*-
"""
Data Retention Service.

CCEA Phase 8 Implementation.

This service manages data retention policies:
    - Configurable retention per tenant
    - Auto-purge based on retention rules
    - Compliance tracking
    - Retention audit logging

Design Doc Reference:
    - Phase 8 (13.3): "Retention per tenant, auto-purge"
    - Data lifecycle management

CLOUD ZONE ONLY.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, Final, List, Optional, Set
from uuid import uuid4


# ============================================================================
# Constants
# ============================================================================

# Default retention periods (days)
DEFAULT_RETENTION_PERIODS: Final[Dict[str, int]] = {
    "telemetry_events": 90,
    "alerts": 365,
    "commands": 180,
    "approval_records": 365 * 7,  # 7 years for compliance
    "access_audits": 365 * 7,  # 7 years for compliance
    "config_blobs": 365,
    "run_data": 365,
    "deployment_data": 365,
}

# Minimum retention periods (cannot go below these)
MIN_RETENTION_PERIODS: Final[Dict[str, int]] = {
    "telemetry_events": 7,
    "alerts": 30,
    "commands": 30,
    "approval_records": 365,  # Minimum 1 year for compliance
    "access_audits": 365,  # Minimum 1 year for compliance
    "config_blobs": 30,
    "run_data": 30,
    "deployment_data": 30,
}


class RetentionAction(Enum):
    """Actions for data after retention period."""

    DELETE = auto()  # Permanently delete
    ARCHIVE = auto()  # Move to cold storage
    ANONYMIZE = auto()  # Anonymize but keep
    AGGREGATE = auto()  # Convert to aggregated form


@dataclass
class RetentionPolicy:
    """
    Retention policy for a data type.

    Attributes:
        id: Policy identifier
        workspace_id: Workspace scope
        data_type: Type of data this policy applies to
        retention_days: Number of days to retain
        action: What to do after retention period
        enabled: Whether policy is active
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    data_type: str = ""
    retention_days: int = 90
    action: RetentionAction = RetentionAction.DELETE
    enabled: bool = True

    # Compliance
    compliance_reason: str = ""
    legal_hold: bool = False
    legal_hold_until: Optional[datetime] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_applied_at: Optional[datetime] = None

    def __post_init__(self):
        """Validate retention period."""
        min_days = MIN_RETENTION_PERIODS.get(self.data_type, 7)
        if self.retention_days < min_days:
            self.retention_days = min_days

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "workspace_id": self.workspace_id,
            "data_type": self.data_type,
            "retention_days": self.retention_days,
            "action": self.action.name,
            "enabled": self.enabled,
            "compliance_reason": self.compliance_reason,
            "legal_hold": self.legal_hold,
            "legal_hold_until": (
                self.legal_hold_until.isoformat() if self.legal_hold_until else None
            ),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_applied_at": self.last_applied_at.isoformat() if self.last_applied_at else None,
        }

    @property
    def cutoff_date(self) -> datetime:
        """Get the cutoff date for retention."""
        return datetime.utcnow() - timedelta(days=self.retention_days)

    @property
    def is_on_legal_hold(self) -> bool:
        """Check if data is on legal hold."""
        if not self.legal_hold:
            return False
        if self.legal_hold_until and datetime.utcnow() > self.legal_hold_until:
            return False
        return True


@dataclass
class RetentionConfig:
    """Global retention configuration."""

    auto_purge_enabled: bool = True
    purge_batch_size: int = 1000
    purge_interval_hours: int = 24
    dry_run_mode: bool = False
    notify_before_purge: bool = True
    notification_days: int = 7


@dataclass
class PurgeResult:
    """Result of a purge operation."""

    success: bool = True
    workspace_id: str = ""
    data_type: str = ""
    records_purged: int = 0
    records_archived: int = 0
    records_anonymized: int = 0
    cutoff_date: Optional[datetime] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None
    was_dry_run: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "workspace_id": self.workspace_id,
            "data_type": self.data_type,
            "records_purged": self.records_purged,
            "records_archived": self.records_archived,
            "records_anonymized": self.records_anonymized,
            "cutoff_date": self.cutoff_date.isoformat() if self.cutoff_date else None,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "was_dry_run": self.was_dry_run,
        }


class RetentionService:
    """
    Data retention and auto-purge service.

    Manages data lifecycle based on retention policies.

    Usage:
        service = RetentionService(purger=my_purger)

        # Create policy
        policy = service.create_policy(
            workspace_id="workspace-123",
            data_type="telemetry_events",
            retention_days=30,
        )

        # Run purge
        results = await service.run_purge()

        # Set legal hold
        service.set_legal_hold("workspace-123", "telemetry_events", reason="Investigation")

    COMPLIANCE:
        - Minimum retention periods enforced
        - Legal hold prevents purging
        - All purge operations logged
    """

    def __init__(
        self,
        config: Optional[RetentionConfig] = None,
        purger: Optional[Callable[[str, str, datetime, RetentionAction], int]] = None,
    ):
        """
        Initialize retention service.

        Args:
            config: Global configuration
            purger: Function to purge data (workspace_id, data_type, cutoff, action) -> count
        """
        self.config = config or RetentionConfig()
        self._purger = purger
        self._policies: Dict[str, Dict[str, RetentionPolicy]] = (
            {}
        )  # workspace_id -> data_type -> policy
        self._audit_log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def create_policy(
        self,
        workspace_id: str,
        data_type: str,
        retention_days: Optional[int] = None,
        action: RetentionAction = RetentionAction.DELETE,
    ) -> RetentionPolicy:
        """
        Create retention policy.

        Args:
            workspace_id: Workspace scope
            data_type: Type of data
            retention_days: Retention period (uses default if not specified)
            action: Action after retention

        Returns:
            Created policy
        """
        if retention_days is None:
            retention_days = DEFAULT_RETENTION_PERIODS.get(data_type, 90)

        policy = RetentionPolicy(
            workspace_id=workspace_id,
            data_type=data_type,
            retention_days=retention_days,
            action=action,
        )

        with self._lock:
            if workspace_id not in self._policies:
                self._policies[workspace_id] = {}
            self._policies[workspace_id][data_type] = policy
            self._log_audit("policy_created", policy)

        return policy

    def get_policy(
        self,
        workspace_id: str,
        data_type: str,
    ) -> Optional[RetentionPolicy]:
        """Get policy for workspace and data type."""
        with self._lock:
            workspace_policies = self._policies.get(workspace_id, {})
            return workspace_policies.get(data_type)

    def get_workspace_policies(self, workspace_id: str) -> List[RetentionPolicy]:
        """Get all policies for a workspace."""
        with self._lock:
            return list(self._policies.get(workspace_id, {}).values())

    def update_policy(
        self,
        workspace_id: str,
        data_type: str,
        retention_days: Optional[int] = None,
        action: Optional[RetentionAction] = None,
        enabled: Optional[bool] = None,
    ) -> Optional[RetentionPolicy]:
        """
        Update existing policy.

        Args:
            workspace_id: Workspace scope
            data_type: Type of data
            retention_days: New retention period
            action: New action
            enabled: Enable/disable

        Returns:
            Updated policy or None
        """
        with self._lock:
            policy = self._policies.get(workspace_id, {}).get(data_type)
            if not policy:
                return None

            if retention_days is not None:
                policy.retention_days = max(retention_days, MIN_RETENTION_PERIODS.get(data_type, 7))
            if action is not None:
                policy.action = action
            if enabled is not None:
                policy.enabled = enabled

            policy.updated_at = datetime.utcnow()
            self._log_audit("policy_updated", policy)

            return policy

    def set_legal_hold(
        self,
        workspace_id: str,
        data_type: str,
        reason: str,
        until: Optional[datetime] = None,
    ) -> bool:
        """
        Set legal hold on data.

        Args:
            workspace_id: Workspace scope
            data_type: Type of data
            reason: Reason for hold
            until: Hold until date (None = indefinite)

        Returns:
            True if hold set
        """
        with self._lock:
            policy = self._policies.get(workspace_id, {}).get(data_type)
            if not policy:
                # Create policy with legal hold
                policy = RetentionPolicy(
                    workspace_id=workspace_id,
                    data_type=data_type,
                    retention_days=DEFAULT_RETENTION_PERIODS.get(data_type, 90),
                    legal_hold=True,
                    legal_hold_until=until,
                    compliance_reason=reason,
                )
                if workspace_id not in self._policies:
                    self._policies[workspace_id] = {}
                self._policies[workspace_id][data_type] = policy
            else:
                policy.legal_hold = True
                policy.legal_hold_until = until
                policy.compliance_reason = reason
                policy.updated_at = datetime.utcnow()

            self._log_audit("legal_hold_set", policy, {"reason": reason})
            return True

    def release_legal_hold(
        self,
        workspace_id: str,
        data_type: str,
        released_by: str,
    ) -> bool:
        """
        Release legal hold on data.

        Args:
            workspace_id: Workspace scope
            data_type: Type of data
            released_by: Who released the hold

        Returns:
            True if hold released
        """
        with self._lock:
            policy = self._policies.get(workspace_id, {}).get(data_type)
            if not policy:
                return False

            policy.legal_hold = False
            policy.legal_hold_until = None
            policy.updated_at = datetime.utcnow()

            self._log_audit("legal_hold_released", policy, {"released_by": released_by})
            return True

    async def run_purge(
        self,
        workspace_id: Optional[str] = None,
    ) -> List[PurgeResult]:
        """
        Run purge for all eligible data.

        Args:
            workspace_id: Limit to specific workspace

        Returns:
            List of purge results
        """
        results = []

        with self._lock:
            policies_to_process = []
            if workspace_id:
                policies_to_process = list(self._policies.get(workspace_id, {}).values())
            else:
                for ws_policies in self._policies.values():
                    policies_to_process.extend(ws_policies.values())

        for policy in policies_to_process:
            result = await self._purge_policy(policy)
            results.append(result)

        return results

    async def _purge_policy(self, policy: RetentionPolicy) -> PurgeResult:
        """Purge data according to policy."""
        start_time = datetime.utcnow()

        # Check if purge should proceed
        if not policy.enabled:
            return PurgeResult(
                success=True,
                workspace_id=policy.workspace_id,
                data_type=policy.data_type,
                error="Policy disabled",
            )

        if policy.is_on_legal_hold:
            return PurgeResult(
                success=True,
                workspace_id=policy.workspace_id,
                data_type=policy.data_type,
                error="Data on legal hold",
            )

        if not self._purger:
            return PurgeResult(
                success=False,
                workspace_id=policy.workspace_id,
                data_type=policy.data_type,
                error="Purger not configured",
            )

        try:
            if self.config.dry_run_mode:
                # In dry run mode, don't actually purge
                count = 0  # Would be estimated count
                result = PurgeResult(
                    success=True,
                    workspace_id=policy.workspace_id,
                    data_type=policy.data_type,
                    cutoff_date=policy.cutoff_date,
                    was_dry_run=True,
                )
            else:
                # Perform actual purge
                count = self._purger(
                    policy.workspace_id,
                    policy.data_type,
                    policy.cutoff_date,
                    policy.action,
                )

                result = PurgeResult(
                    success=True,
                    workspace_id=policy.workspace_id,
                    data_type=policy.data_type,
                    cutoff_date=policy.cutoff_date,
                )

                if policy.action == RetentionAction.DELETE:
                    result.records_purged = count
                elif policy.action == RetentionAction.ARCHIVE:
                    result.records_archived = count
                elif policy.action == RetentionAction.ANONYMIZE:
                    result.records_anonymized = count

            # Update policy
            with self._lock:
                policy.last_applied_at = datetime.utcnow()

            result.duration_seconds = (datetime.utcnow() - start_time).total_seconds()

            self._log_audit(
                "purge_completed",
                policy,
                {
                    "records_purged": result.records_purged,
                    "records_archived": result.records_archived,
                    "records_anonymized": result.records_anonymized,
                },
            )

            return result

        except Exception as e:
            return PurgeResult(
                success=False,
                workspace_id=policy.workspace_id,
                data_type=policy.data_type,
                error=str(e),
                duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
            )

    def get_retention_report(self, workspace_id: str) -> Dict[str, Any]:
        """
        Get retention status report for workspace.

        Args:
            workspace_id: Workspace to report on

        Returns:
            Report dictionary
        """
        policies = self.get_workspace_policies(workspace_id)

        report = {
            "workspace_id": workspace_id,
            "generated_at": datetime.utcnow().isoformat(),
            "policies": [],
            "legal_holds": [],
            "upcoming_purges": [],
        }

        for policy in policies:
            report["policies"].append(
                {
                    "data_type": policy.data_type,
                    "retention_days": policy.retention_days,
                    "action": policy.action.name,
                    "enabled": policy.enabled,
                    "cutoff_date": policy.cutoff_date.isoformat(),
                    "last_applied": (
                        policy.last_applied_at.isoformat() if policy.last_applied_at else None
                    ),
                }
            )

            if policy.is_on_legal_hold:
                report["legal_holds"].append(
                    {
                        "data_type": policy.data_type,
                        "reason": policy.compliance_reason,
                        "until": (
                            policy.legal_hold_until.isoformat()
                            if policy.legal_hold_until
                            else "indefinite"
                        ),
                    }
                )

            if policy.enabled and not policy.is_on_legal_hold:
                report["upcoming_purges"].append(
                    {
                        "data_type": policy.data_type,
                        "cutoff_date": policy.cutoff_date.isoformat(),
                        "action": policy.action.name,
                    }
                )

        return report

    def _log_audit(
        self,
        action: str,
        policy: RetentionPolicy,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log audit event."""
        entry = {
            "action": action,
            "policy_id": policy.id,
            "workspace_id": policy.workspace_id,
            "data_type": policy.data_type,
            "timestamp": datetime.utcnow().isoformat(),
            **(extra or {}),
        }
        self._audit_log.append(entry)

    def get_audit_log(self, workspace_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get audit log."""
        with self._lock:
            if workspace_id:
                return [e for e in self._audit_log if e.get("workspace_id") == workspace_id]
            return list(self._audit_log)
