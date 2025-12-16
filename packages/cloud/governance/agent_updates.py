"""
Agent Update Service - Phase 7 GDPR Implementation

Implements Design Doc Section 15.2 agent update controls:
- Signed agent updates
- Staged rollout
- Rollback support
- Enterprise version pinning and change windows

Reference: docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L917-L925
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4


# =============================================================================
# Enums and Constants
# =============================================================================

class UpdateChannel(str, Enum):
    """Update channels for agent releases."""
    STABLE = "stable"
    BETA = "beta"
    CANARY = "canary"
    ENTERPRISE = "enterprise"


class RolloutStatus(str, Enum):
    """Status of a rollout stage or overall rollout."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class UpdateResult(str, Enum):
    """Result of an individual agent update."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    REJECTED = "rejected"
    SKIPPED = "skipped"


class UpdateEventType(str, Enum):
    """Types of update events."""
    UPDATE_PUBLISHED = "update_published"
    ROLLOUT_STARTED = "rollout_started"
    ROLLOUT_STAGE_STARTED = "rollout_stage_started"
    ROLLOUT_STAGE_COMPLETED = "rollout_stage_completed"
    ROLLOUT_PAUSED = "rollout_paused"
    ROLLOUT_RESUMED = "rollout_resumed"
    ROLLOUT_COMPLETED = "rollout_completed"
    ROLLOUT_FAILED = "rollout_failed"
    ROLLOUT_CANCELLED = "rollout_cancelled"
    ROLLBACK_STARTED = "rollback_started"
    ROLLBACK_COMPLETED = "rollback_completed"
    ROLLBACK_FAILED = "rollback_failed"
    AGENT_UPDATED = "agent_updated"
    AGENT_UPDATE_FAILED = "agent_update_failed"
    VERSION_PINNED = "version_pinned"
    VERSION_UNPINNED = "version_unpinned"
    CHANGE_WINDOW_SET = "change_window_set"
    UPDATE_BLOCKED = "update_blocked"
    SIGNATURE_VERIFIED = "signature_verified"
    SIGNATURE_FAILED = "signature_failed"


# Constants
SEMVER_PATTERN = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.-]+))?(?:\+([a-zA-Z0-9.-]+))?$")
DEFAULT_STAGE_TIMEOUT_HOURS = 24
MAX_STAGE_TIMEOUT_HOURS = 168  # 1 week
DEFAULT_ERROR_RATE_THRESHOLD = 0.05  # 5%
CRITICAL_ERROR_RATE_THRESHOLD = 0.10  # 10%
MIN_STAGE_WAIT_MINUTES = 15


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AgentUpdate:
    """An agent update package."""
    update_id: str = field(default_factory=lambda: str(uuid4()))
    version: str = ""  # semver
    channel: UpdateChannel = UpdateChannel.STABLE
    artifact_digest: str = ""
    signature: str = ""
    signer_id: str = ""
    release_notes: str = ""
    min_agent_version: Optional[str] = None  # Compatibility
    max_agent_version: Optional[str] = None
    deprecated_version: Optional[str] = None  # Version being deprecated
    breaking_changes: bool = False
    published_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    published_by: str = ""
    size_bytes: int = 0
    checksum: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.version and not SEMVER_PATTERN.match(self.version):
            raise ValueError(f"Invalid semver format: {self.version}")

    def is_compatible_with(self, current_version: str) -> bool:
        """Check if this update is compatible with the current version."""
        if not current_version:
            return True

        if self.min_agent_version:
            if self._compare_versions(current_version, self.min_agent_version) < 0:
                return False

        if self.max_agent_version:
            if self._compare_versions(current_version, self.max_agent_version) > 0:
                return False

        return True

    @staticmethod
    def _compare_versions(v1: str, v2: str) -> int:
        """Compare two semver versions. Returns -1, 0, or 1."""
        m1 = SEMVER_PATTERN.match(v1)
        m2 = SEMVER_PATTERN.match(v2)

        if not m1 or not m2:
            return 0

        for i in range(1, 4):
            n1, n2 = int(m1.group(i)), int(m2.group(i))
            if n1 < n2:
                return -1
            if n1 > n2:
                return 1

        return 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "update_id": self.update_id,
            "version": self.version,
            "channel": self.channel.value,
            "artifact_digest": self.artifact_digest,
            "signature": self.signature,
            "signer_id": self.signer_id,
            "release_notes": self.release_notes,
            "min_agent_version": self.min_agent_version,
            "max_agent_version": self.max_agent_version,
            "deprecated_version": self.deprecated_version,
            "breaking_changes": self.breaking_changes,
            "published_at": self.published_at.isoformat(),
            "published_by": self.published_by,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "metadata": self.metadata,
        }


@dataclass
class RolloutCriteria:
    """Criteria for proceeding with a rollout stage."""
    min_success_rate: float = 0.95  # 95% success required
    max_error_rate: float = DEFAULT_ERROR_RATE_THRESHOLD
    min_agents_updated: int = 1
    min_wait_minutes: int = MIN_STAGE_WAIT_MINUTES
    require_manual_approval: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_success_rate": self.min_success_rate,
            "max_error_rate": self.max_error_rate,
            "min_agents_updated": self.min_agents_updated,
            "min_wait_minutes": self.min_wait_minutes,
            "require_manual_approval": self.require_manual_approval,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
        }


@dataclass
class RolloutStageMetrics:
    """Metrics for a rollout stage."""
    agents_targeted: int = 0
    agents_updated: int = 0
    agents_failed: int = 0
    agents_skipped: int = 0
    success_rate: float = 0.0
    error_rate: float = 0.0
    average_update_time_seconds: float = 0.0
    errors_by_type: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agents_targeted": self.agents_targeted,
            "agents_updated": self.agents_updated,
            "agents_failed": self.agents_failed,
            "agents_skipped": self.agents_skipped,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
            "average_update_time_seconds": self.average_update_time_seconds,
            "errors_by_type": self.errors_by_type,
        }


@dataclass
class RolloutStage:
    """A stage in a staged rollout."""
    stage_id: str = field(default_factory=lambda: str(uuid4()))
    stage_name: str = ""  # e.g., "canary", "early_adopter", "general"
    stage_order: int = 0
    percentage: int = 0  # 0-100
    target_workspaces: Set[str] = field(default_factory=set)  # Empty = random selection
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    criteria: RolloutCriteria = field(default_factory=RolloutCriteria)
    metrics: RolloutStageMetrics = field(default_factory=RolloutStageMetrics)
    status: RolloutStatus = RolloutStatus.PENDING
    error: Optional[str] = None

    def is_criteria_met(self) -> Tuple[bool, List[str]]:
        """Check if the criteria for proceeding are met."""
        issues = []

        if self.metrics.success_rate < self.criteria.min_success_rate:
            issues.append(f"Success rate {self.metrics.success_rate:.1%} below threshold {self.criteria.min_success_rate:.1%}")

        if self.metrics.error_rate > self.criteria.max_error_rate:
            issues.append(f"Error rate {self.metrics.error_rate:.1%} above threshold {self.criteria.max_error_rate:.1%}")

        if self.metrics.agents_updated < self.criteria.min_agents_updated:
            issues.append(f"Only {self.metrics.agents_updated} agents updated, need {self.criteria.min_agents_updated}")

        if self.started_at:
            elapsed_minutes = (datetime.now(timezone.utc) - self.started_at).total_seconds() / 60
            if elapsed_minutes < self.criteria.min_wait_minutes:
                issues.append(f"Must wait {self.criteria.min_wait_minutes - elapsed_minutes:.0f} more minutes")

        if self.criteria.require_manual_approval and not self.criteria.approved_by:
            issues.append("Manual approval required but not received")

        return len(issues) == 0, issues

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "stage_order": self.stage_order,
            "percentage": self.percentage,
            "target_workspaces": list(self.target_workspaces),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "criteria": self.criteria.to_dict(),
            "metrics": self.metrics.to_dict(),
            "status": self.status.value,
            "error": self.error,
        }


@dataclass
class RolloutPlan:
    """A staged rollout plan."""
    rollout_id: str = field(default_factory=lambda: str(uuid4()))
    update_id: str = ""
    version: str = ""
    channel: UpdateChannel = UpdateChannel.STABLE
    stages: List[RolloutStage] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: RolloutStatus = RolloutStatus.PENDING
    auto_proceed: bool = True  # Auto-advance on success
    pause_on_error_rate: float = CRITICAL_ERROR_RATE_THRESHOLD
    rollback_on_critical: bool = True
    current_stage_index: int = 0
    target_version: str = ""
    previous_version: Optional[str] = None

    def get_current_stage(self) -> Optional[RolloutStage]:
        """Get the current active stage."""
        if 0 <= self.current_stage_index < len(self.stages):
            return self.stages[self.current_stage_index]
        return None

    def get_overall_metrics(self) -> RolloutStageMetrics:
        """Get aggregated metrics across all stages."""
        metrics = RolloutStageMetrics()
        for stage in self.stages:
            metrics.agents_targeted += stage.metrics.agents_targeted
            metrics.agents_updated += stage.metrics.agents_updated
            metrics.agents_failed += stage.metrics.agents_failed
            metrics.agents_skipped += stage.metrics.agents_skipped
            for error_type, count in stage.metrics.errors_by_type.items():
                metrics.errors_by_type[error_type] = metrics.errors_by_type.get(error_type, 0) + count

        total = metrics.agents_updated + metrics.agents_failed
        if total > 0:
            metrics.success_rate = metrics.agents_updated / total
            metrics.error_rate = metrics.agents_failed / total

        return metrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rollout_id": self.rollout_id,
            "update_id": self.update_id,
            "version": self.version,
            "channel": self.channel.value,
            "stages": [s.to_dict() for s in self.stages],
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value,
            "auto_proceed": self.auto_proceed,
            "pause_on_error_rate": self.pause_on_error_rate,
            "rollback_on_critical": self.rollback_on_critical,
            "current_stage_index": self.current_stage_index,
            "target_version": self.target_version,
            "previous_version": self.previous_version,
        }


@dataclass
class RollbackRequest:
    """A rollback request."""
    rollback_id: str = field(default_factory=lambda: str(uuid4()))
    rollout_id: str = ""
    from_version: str = ""
    to_version: str = ""
    reason: str = ""
    requested_by: str = ""
    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: RolloutStatus = RolloutStatus.PENDING
    affected_agents: List[str] = field(default_factory=list)
    agents_rolled_back: int = 0
    agents_failed: int = 0
    error: Optional[str] = None
    evidence_hash: str = ""

    def __post_init__(self) -> None:
        if not self.evidence_hash:
            self.evidence_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = json.dumps({
            "rollback_id": self.rollback_id,
            "rollout_id": self.rollout_id,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "requested_at": self.requested_at.isoformat(),
            "requested_by": self.requested_by,
        }, sort_keys=True)
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rollback_id": self.rollback_id,
            "rollout_id": self.rollout_id,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "reason": self.reason,
            "requested_by": self.requested_by,
            "requested_at": self.requested_at.isoformat(),
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value,
            "affected_agents": self.affected_agents,
            "agents_rolled_back": self.agents_rolled_back,
            "agents_failed": self.agents_failed,
            "error": self.error,
            "evidence_hash": self.evidence_hash,
        }


@dataclass
class ChangeWindow:
    """Change window for enterprise updates."""
    allowed_days: Set[int] = field(default_factory=lambda: {0, 1, 2, 3, 4})  # Mon-Fri
    start_hour: int = 2  # UTC hour (0-23)
    end_hour: int = 6  # UTC hour (0-23)
    timezone_name: str = "UTC"
    exclude_dates: Set[str] = field(default_factory=set)  # ISO dates to exclude

    def is_within_window(self, dt: Optional[datetime] = None) -> bool:
        """Check if the given datetime is within the change window."""
        if dt is None:
            dt = datetime.now(timezone.utc)

        # Check day of week
        if dt.weekday() not in self.allowed_days:
            return False

        # Check time
        hour = dt.hour
        if self.start_hour <= self.end_hour:
            # Normal range (e.g., 2-6)
            if not (self.start_hour <= hour < self.end_hour):
                return False
        else:
            # Overnight range (e.g., 22-6)
            if not (hour >= self.start_hour or hour < self.end_hour):
                return False

        # Check excluded dates
        date_str = dt.strftime("%Y-%m-%d")
        if date_str in self.exclude_dates:
            return False

        return True

    def next_window_start(self, after: Optional[datetime] = None) -> datetime:
        """Get the next change window start time."""
        if after is None:
            after = datetime.now(timezone.utc)

        current = after.replace(minute=0, second=0, microsecond=0)

        for _ in range(14):  # Check up to 2 weeks ahead
            if current.weekday() in self.allowed_days:
                window_start = current.replace(hour=self.start_hour)
                if window_start > after:
                    date_str = window_start.strftime("%Y-%m-%d")
                    if date_str not in self.exclude_dates:
                        return window_start

            current = current + timedelta(days=1)

        # Fallback
        return after + timedelta(days=7)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed_days": list(self.allowed_days),
            "start_hour": self.start_hour,
            "end_hour": self.end_hour,
            "timezone_name": self.timezone_name,
            "exclude_dates": list(self.exclude_dates),
        }


@dataclass
class VersionPin:
    """Enterprise version pinning."""
    pin_id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    pinned_version: str = ""
    channel: UpdateChannel = UpdateChannel.ENTERPRISE
    pinned_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    pinned_by: str = ""
    reason: str = ""
    expires_at: Optional[datetime] = None
    change_window: Optional[ChangeWindow] = None
    active: bool = True

    def is_expired(self) -> bool:
        """Check if the pin has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) >= self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pin_id": self.pin_id,
            "workspace_id": self.workspace_id,
            "pinned_version": self.pinned_version,
            "channel": self.channel.value,
            "pinned_at": self.pinned_at.isoformat(),
            "pinned_by": self.pinned_by,
            "reason": self.reason,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "change_window": self.change_window.to_dict() if self.change_window else None,
            "active": self.active,
        }


@dataclass
class AgentUpdateRecord:
    """Record of an individual agent update."""
    record_id: str = field(default_factory=lambda: str(uuid4()))
    agent_id: str = ""
    workspace_id: str = ""
    rollout_id: Optional[str] = None
    from_version: str = ""
    to_version: str = ""
    result: UpdateResult = UpdateResult.SUCCESS
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None
    signature_verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "agent_id": self.agent_id,
            "workspace_id": self.workspace_id,
            "rollout_id": self.rollout_id,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "result": self.result.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "signature_verified": self.signature_verified,
        }


@dataclass
class UpdateEvent:
    """Event in the update audit log."""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    workspace_id: str = ""
    event_type: UpdateEventType = UpdateEventType.UPDATE_PUBLISHED
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    actor_id: str = ""
    rollout_id: Optional[str] = None
    update_id: Optional[str] = None
    agent_id: Optional[str] = None
    version: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    result: str = "success"
    integrity_hash: str = ""

    def __post_init__(self) -> None:
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = json.dumps({
            "event_id": self.event_id,
            "workspace_id": self.workspace_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "actor_id": self.actor_id,
            "rollout_id": self.rollout_id,
            "version": self.version,
            "result": self.result,
        }, sort_keys=True)
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "workspace_id": self.workspace_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "actor_id": self.actor_id,
            "rollout_id": self.rollout_id,
            "update_id": self.update_id,
            "agent_id": self.agent_id,
            "version": self.version,
            "details": self.details,
            "result": self.result,
            "integrity_hash": self.integrity_hash,
        }


@dataclass
class AgentUpdateStats:
    """Statistics for agent updates."""
    workspace_id: str = ""
    total_updates_published: int = 0
    total_rollouts: int = 0
    rollouts_completed: int = 0
    rollouts_failed: int = 0
    rollouts_rolled_back: int = 0
    agents_updated_30d: int = 0
    agents_failed_30d: int = 0
    success_rate_30d: float = 0.0
    average_rollout_duration_hours: float = 0.0
    version_pins_active: int = 0
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "total_updates_published": self.total_updates_published,
            "total_rollouts": self.total_rollouts,
            "rollouts_completed": self.rollouts_completed,
            "rollouts_failed": self.rollouts_failed,
            "rollouts_rolled_back": self.rollouts_rolled_back,
            "agents_updated_30d": self.agents_updated_30d,
            "agents_failed_30d": self.agents_failed_30d,
            "success_rate_30d": self.success_rate_30d,
            "average_rollout_duration_hours": self.average_rollout_duration_hours,
            "version_pins_active": self.version_pins_active,
            "evaluated_at": self.evaluated_at.isoformat(),
        }


# =============================================================================
# Agent Update Service
# =============================================================================

class AgentUpdateService:
    """
    Agent Update Service.

    Manages:
    - Signed agent update packages
    - Staged rollout with configurable stages
    - Rollback procedures
    - Enterprise version pinning and change windows

    Reference: Design Doc Section 15.2
    """

    def __init__(
        self,
        require_signatures: bool = True,
        default_rollout_stages: Optional[List[Dict[str, Any]]] = None,
        on_event: Optional[Callable[[UpdateEvent], None]] = None,
        on_rollout_complete: Optional[Callable[[RolloutPlan], None]] = None,
        on_rollback: Optional[Callable[[RollbackRequest], None]] = None,
    ):
        """
        Initialize Agent Update Service.

        Args:
            require_signatures: Require all updates to be signed
            default_rollout_stages: Default stages for rollouts
            on_event: Callback for update events
            on_rollout_complete: Callback when rollout completes
            on_rollback: Callback when rollback occurs
        """
        self._lock = threading.RLock()  # Re-entrant lock for nested calls

        # Configuration
        self._require_signatures = require_signatures
        self._default_stages = default_rollout_stages or [
            {"name": "canary", "percentage": 1},
            {"name": "early_adopter", "percentage": 10},
            {"name": "general", "percentage": 100},
        ]

        # Callbacks
        self._on_event = on_event
        self._on_rollout_complete = on_rollout_complete
        self._on_rollback = on_rollback

        # Storage
        self._updates: Dict[str, AgentUpdate] = {}  # update_id -> update
        self._updates_by_version: Dict[str, str] = {}  # version -> update_id
        self._rollouts: Dict[str, RolloutPlan] = {}  # rollout_id -> plan
        self._rollbacks: Dict[str, RollbackRequest] = {}  # rollback_id -> request
        self._version_pins: Dict[str, VersionPin] = {}  # pin_id -> pin
        self._workspace_pins: Dict[str, str] = {}  # workspace_id -> pin_id
        self._update_records: List[AgentUpdateRecord] = []
        self._events: List[UpdateEvent] = []

    # =========================================================================
    # Update Package Management
    # =========================================================================

    def publish_update(
        self,
        version: str,
        artifact_digest: str,
        published_by: str,
        channel: UpdateChannel = UpdateChannel.STABLE,
        signature: Optional[str] = None,
        signer_id: Optional[str] = None,
        release_notes: str = "",
        min_agent_version: Optional[str] = None,
        max_agent_version: Optional[str] = None,
        breaking_changes: bool = False,
        size_bytes: int = 0,
        checksum: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentUpdate:
        """Publish a new agent update."""
        if not SEMVER_PATTERN.match(version):
            raise ValueError(f"Invalid semver format: {version}")

        if self._require_signatures and not signature:
            raise ValueError("Update signature is required")

        # Check version doesn't already exist
        with self._lock:
            if version in self._updates_by_version:
                raise ValueError(f"Version {version} already exists")

        update = AgentUpdate(
            version=version,
            channel=channel,
            artifact_digest=artifact_digest,
            signature=signature or "",
            signer_id=signer_id or "",
            release_notes=release_notes,
            min_agent_version=min_agent_version,
            max_agent_version=max_agent_version,
            breaking_changes=breaking_changes,
            published_by=published_by,
            size_bytes=size_bytes,
            checksum=checksum,
            metadata=metadata or {},
        )

        with self._lock:
            self._updates[update.update_id] = update
            self._updates_by_version[version] = update.update_id

        # Log event
        event = UpdateEvent(
            event_type=UpdateEventType.UPDATE_PUBLISHED,
            actor_id=published_by,
            update_id=update.update_id,
            version=version,
            details={
                "channel": channel.value,
                "breaking_changes": breaking_changes,
                "signed": bool(signature),
            },
        )
        self._log_event(event)

        return update

    def get_update(self, update_id: str) -> Optional[AgentUpdate]:
        """Get update by ID."""
        with self._lock:
            return self._updates.get(update_id)

    def get_update_by_version(self, version: str) -> Optional[AgentUpdate]:
        """Get update by version."""
        with self._lock:
            update_id = self._updates_by_version.get(version)
            if update_id:
                return self._updates.get(update_id)
        return None

    def list_updates(
        self,
        channel: Optional[UpdateChannel] = None,
        limit: int = 100,
    ) -> List[AgentUpdate]:
        """List updates with optional filtering."""
        with self._lock:
            updates = list(self._updates.values())

        if channel:
            updates = [u for u in updates if u.channel == channel]

        # Sort by published_at descending
        updates.sort(key=lambda u: u.published_at, reverse=True)
        return updates[:limit]

    def get_latest_update(
        self,
        channel: UpdateChannel = UpdateChannel.STABLE,
    ) -> Optional[AgentUpdate]:
        """Get the latest update for a channel."""
        updates = self.list_updates(channel=channel, limit=1)
        return updates[0] if updates else None

    # =========================================================================
    # Staged Rollout
    # =========================================================================

    def create_rollout(
        self,
        update_id: str,
        created_by: str,
        stages: Optional[List[Dict[str, Any]]] = None,
        auto_proceed: bool = True,
        pause_on_error_rate: float = CRITICAL_ERROR_RATE_THRESHOLD,
        rollback_on_critical: bool = True,
    ) -> RolloutPlan:
        """Create a staged rollout plan."""
        update = self.get_update(update_id)
        if not update:
            raise ValueError(f"Update {update_id} not found")

        stage_configs = stages or self._default_stages
        rollout_stages = []

        for i, config in enumerate(stage_configs):
            criteria = RolloutCriteria(
                min_success_rate=config.get("min_success_rate", 0.95),
                max_error_rate=config.get("max_error_rate", DEFAULT_ERROR_RATE_THRESHOLD),
                min_agents_updated=config.get("min_agents_updated", 1),
                min_wait_minutes=config.get("min_wait_minutes", MIN_STAGE_WAIT_MINUTES),
                require_manual_approval=config.get("require_manual_approval", False),
            )

            stage = RolloutStage(
                stage_name=config.get("name", f"stage_{i}"),
                stage_order=i,
                percentage=config.get("percentage", 100),
                target_workspaces=set(config.get("target_workspaces", [])),
                criteria=criteria,
            )
            rollout_stages.append(stage)

        plan = RolloutPlan(
            update_id=update_id,
            version=update.version,
            channel=update.channel,
            stages=rollout_stages,
            created_by=created_by,
            auto_proceed=auto_proceed,
            pause_on_error_rate=pause_on_error_rate,
            rollback_on_critical=rollback_on_critical,
            target_version=update.version,
        )

        with self._lock:
            self._rollouts[plan.rollout_id] = plan

        return plan

    def start_rollout(
        self,
        rollout_id: str,
        started_by: str,
    ) -> RolloutPlan:
        """Start a rollout."""
        with self._lock:
            plan = self._rollouts.get(rollout_id)
            if not plan:
                raise ValueError(f"Rollout {rollout_id} not found")

            if plan.status != RolloutStatus.PENDING:
                raise ValueError(f"Rollout is in {plan.status.value} status, cannot start")

            plan.status = RolloutStatus.IN_PROGRESS
            plan.started_at = datetime.now(timezone.utc)

            # Start first stage
            if plan.stages:
                first_stage = plan.stages[0]
                first_stage.status = RolloutStatus.IN_PROGRESS
                first_stage.started_at = datetime.now(timezone.utc)

        # Log event
        event = UpdateEvent(
            event_type=UpdateEventType.ROLLOUT_STARTED,
            actor_id=started_by,
            rollout_id=rollout_id,
            version=plan.version,
            details={
                "stages_count": len(plan.stages),
                "first_stage": plan.stages[0].stage_name if plan.stages else None,
            },
        )
        self._log_event(event)

        return plan

    def record_agent_update(
        self,
        rollout_id: str,
        agent_id: str,
        workspace_id: str,
        from_version: str,
        to_version: str,
        result: UpdateResult,
        duration_seconds: float = 0.0,
        error: Optional[str] = None,
        signature_verified: bool = False,
    ) -> AgentUpdateRecord:
        """Record an individual agent update result."""
        record = AgentUpdateRecord(
            agent_id=agent_id,
            workspace_id=workspace_id,
            rollout_id=rollout_id,
            from_version=from_version,
            to_version=to_version,
            result=result,
            completed_at=datetime.now(timezone.utc),
            duration_seconds=duration_seconds,
            error=error,
            signature_verified=signature_verified,
        )

        with self._lock:
            self._update_records.append(record)

            # Update stage metrics
            plan = self._rollouts.get(rollout_id)
            if plan:
                stage = plan.get_current_stage()
                if stage:
                    stage.metrics.agents_targeted += 1
                    if result == UpdateResult.SUCCESS:
                        stage.metrics.agents_updated += 1
                    elif result == UpdateResult.FAILED:
                        stage.metrics.agents_failed += 1
                        if error:
                            error_type = error.split(":")[0] if ":" in error else "unknown"
                            stage.metrics.errors_by_type[error_type] = stage.metrics.errors_by_type.get(error_type, 0) + 1
                    elif result == UpdateResult.SKIPPED:
                        stage.metrics.agents_skipped += 1

                    # Recalculate rates
                    total = stage.metrics.agents_updated + stage.metrics.agents_failed
                    if total > 0:
                        stage.metrics.success_rate = stage.metrics.agents_updated / total
                        stage.metrics.error_rate = stage.metrics.agents_failed / total

                    # Check for critical error rate
                    if stage.metrics.error_rate >= plan.pause_on_error_rate:
                        plan.status = RolloutStatus.PAUSED
                        stage.status = RolloutStatus.PAUSED
                        stage.error = f"Error rate {stage.metrics.error_rate:.1%} exceeded threshold"

        # Log event
        event_type = UpdateEventType.AGENT_UPDATED if result == UpdateResult.SUCCESS else UpdateEventType.AGENT_UPDATE_FAILED
        event = UpdateEvent(
            workspace_id=workspace_id,
            event_type=event_type,
            actor_id="system",
            rollout_id=rollout_id,
            agent_id=agent_id,
            version=to_version,
            details={
                "from_version": from_version,
                "result": result.value,
                "duration_seconds": duration_seconds,
            },
            result="success" if result == UpdateResult.SUCCESS else "failure",
        )
        self._log_event(event)

        return record

    def advance_stage(
        self,
        rollout_id: str,
        advanced_by: str,
        force: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """
        Advance to the next rollout stage.

        Returns: (success, error_message)
        """
        with self._lock:
            plan = self._rollouts.get(rollout_id)
            if not plan:
                return False, f"Rollout {rollout_id} not found"

            if plan.status not in (RolloutStatus.IN_PROGRESS, RolloutStatus.PAUSED):
                return False, f"Rollout is in {plan.status.value} status"

            current_stage = plan.get_current_stage()
            if not current_stage:
                return False, "No current stage"

            # Check criteria unless forcing
            if not force:
                criteria_met, issues = current_stage.is_criteria_met()
                if not criteria_met:
                    return False, f"Criteria not met: {'; '.join(issues)}"

            # Complete current stage
            current_stage.status = RolloutStatus.COMPLETED
            current_stage.completed_at = datetime.now(timezone.utc)

            # Log stage completion
            event = UpdateEvent(
                event_type=UpdateEventType.ROLLOUT_STAGE_COMPLETED,
                actor_id=advanced_by,
                rollout_id=rollout_id,
                version=plan.version,
                details={
                    "stage_name": current_stage.stage_name,
                    "metrics": current_stage.metrics.to_dict(),
                },
            )
            self._log_event(event)

            # Check if there's a next stage
            plan.current_stage_index += 1
            if plan.current_stage_index >= len(plan.stages):
                # Rollout complete
                plan.status = RolloutStatus.COMPLETED
                plan.completed_at = datetime.now(timezone.utc)

                # Log completion
                completion_event = UpdateEvent(
                    event_type=UpdateEventType.ROLLOUT_COMPLETED,
                    actor_id=advanced_by,
                    rollout_id=rollout_id,
                    version=plan.version,
                    details={"overall_metrics": plan.get_overall_metrics().to_dict()},
                )
                self._log_event(completion_event)

                if self._on_rollout_complete:
                    try:
                        self._on_rollout_complete(plan)
                    except Exception:
                        pass

                return True, None

            # Start next stage
            next_stage = plan.stages[plan.current_stage_index]
            next_stage.status = RolloutStatus.IN_PROGRESS
            next_stage.started_at = datetime.now(timezone.utc)
            plan.status = RolloutStatus.IN_PROGRESS

            # Log stage start
            stage_event = UpdateEvent(
                event_type=UpdateEventType.ROLLOUT_STAGE_STARTED,
                actor_id=advanced_by,
                rollout_id=rollout_id,
                version=plan.version,
                details={
                    "stage_name": next_stage.stage_name,
                    "percentage": next_stage.percentage,
                },
            )
            self._log_event(stage_event)

        return True, None

    def pause_rollout(
        self,
        rollout_id: str,
        paused_by: str,
        reason: str = "",
    ) -> RolloutPlan:
        """Pause a rollout."""
        with self._lock:
            plan = self._rollouts.get(rollout_id)
            if not plan:
                raise ValueError(f"Rollout {rollout_id} not found")

            if plan.status != RolloutStatus.IN_PROGRESS:
                raise ValueError(f"Rollout is in {plan.status.value} status, cannot pause")

            plan.status = RolloutStatus.PAUSED

            current_stage = plan.get_current_stage()
            if current_stage:
                current_stage.status = RolloutStatus.PAUSED
                current_stage.error = reason

        # Log event
        event = UpdateEvent(
            event_type=UpdateEventType.ROLLOUT_PAUSED,
            actor_id=paused_by,
            rollout_id=rollout_id,
            version=plan.version,
            details={"reason": reason},
        )
        self._log_event(event)

        return plan

    def resume_rollout(
        self,
        rollout_id: str,
        resumed_by: str,
    ) -> RolloutPlan:
        """Resume a paused rollout."""
        with self._lock:
            plan = self._rollouts.get(rollout_id)
            if not plan:
                raise ValueError(f"Rollout {rollout_id} not found")

            if plan.status != RolloutStatus.PAUSED:
                raise ValueError(f"Rollout is in {plan.status.value} status, cannot resume")

            plan.status = RolloutStatus.IN_PROGRESS

            current_stage = plan.get_current_stage()
            if current_stage:
                current_stage.status = RolloutStatus.IN_PROGRESS
                current_stage.error = None

        # Log event
        event = UpdateEvent(
            event_type=UpdateEventType.ROLLOUT_RESUMED,
            actor_id=resumed_by,
            rollout_id=rollout_id,
            version=plan.version,
        )
        self._log_event(event)

        return plan

    def cancel_rollout(
        self,
        rollout_id: str,
        cancelled_by: str,
        reason: str = "",
    ) -> RolloutPlan:
        """Cancel a rollout."""
        with self._lock:
            plan = self._rollouts.get(rollout_id)
            if not plan:
                raise ValueError(f"Rollout {rollout_id} not found")

            if plan.status in (RolloutStatus.COMPLETED, RolloutStatus.CANCELLED, RolloutStatus.ROLLED_BACK):
                raise ValueError(f"Rollout is in {plan.status.value} status, cannot cancel")

            plan.status = RolloutStatus.CANCELLED
            plan.completed_at = datetime.now(timezone.utc)

            current_stage = plan.get_current_stage()
            if current_stage and current_stage.status == RolloutStatus.IN_PROGRESS:
                current_stage.status = RolloutStatus.CANCELLED
                current_stage.error = reason

        # Log event
        event = UpdateEvent(
            event_type=UpdateEventType.ROLLOUT_CANCELLED,
            actor_id=cancelled_by,
            rollout_id=rollout_id,
            version=plan.version,
            details={"reason": reason},
        )
        self._log_event(event)

        return plan

    def get_rollout(self, rollout_id: str) -> Optional[RolloutPlan]:
        """Get rollout by ID."""
        with self._lock:
            return self._rollouts.get(rollout_id)

    def list_rollouts(
        self,
        status: Optional[RolloutStatus] = None,
        channel: Optional[UpdateChannel] = None,
        limit: int = 100,
    ) -> List[RolloutPlan]:
        """List rollouts with optional filtering."""
        with self._lock:
            rollouts = list(self._rollouts.values())

        if status:
            rollouts = [r for r in rollouts if r.status == status]
        if channel:
            rollouts = [r for r in rollouts if r.channel == channel]

        rollouts.sort(key=lambda r: r.created_at, reverse=True)
        return rollouts[:limit]

    # =========================================================================
    # Rollback
    # =========================================================================

    def request_rollback(
        self,
        rollout_id: str,
        to_version: str,
        reason: str,
        requested_by: str,
        affected_agents: Optional[List[str]] = None,
    ) -> RollbackRequest:
        """Request a rollback."""
        if len(reason) < 10:
            raise ValueError("Rollback reason must be at least 10 characters")

        plan = self.get_rollout(rollout_id)
        if not plan:
            raise ValueError(f"Rollout {rollout_id} not found")

        rollback = RollbackRequest(
            rollout_id=rollout_id,
            from_version=plan.version,
            to_version=to_version,
            reason=reason,
            requested_by=requested_by,
            affected_agents=affected_agents or [],
        )

        with self._lock:
            self._rollbacks[rollback.rollback_id] = rollback

        # Log event
        event = UpdateEvent(
            event_type=UpdateEventType.ROLLBACK_STARTED,
            actor_id=requested_by,
            rollout_id=rollout_id,
            version=to_version,
            details={
                "from_version": plan.version,
                "reason": reason,
                "affected_agents_count": len(affected_agents) if affected_agents else 0,
            },
        )
        self._log_event(event)

        return rollback

    def approve_rollback(
        self,
        rollback_id: str,
        approved_by: str,
    ) -> RollbackRequest:
        """Approve a rollback request."""
        with self._lock:
            rollback = self._rollbacks.get(rollback_id)
            if not rollback:
                raise ValueError(f"Rollback {rollback_id} not found")

            if rollback.status != RolloutStatus.PENDING:
                raise ValueError(f"Rollback is in {rollback.status.value} status")

            if rollback.requested_by == approved_by:
                raise ValueError("Self-approval is not allowed")

            rollback.approved_by = approved_by
            rollback.approved_at = datetime.now(timezone.utc)

        return rollback

    def execute_rollback(
        self,
        rollback_id: str,
        executed_by: str,
    ) -> RollbackRequest:
        """Execute an approved rollback."""
        with self._lock:
            rollback = self._rollbacks.get(rollback_id)
            if not rollback:
                raise ValueError(f"Rollback {rollback_id} not found")

            if rollback.status != RolloutStatus.PENDING:
                raise ValueError(f"Rollback is in {rollback.status.value} status")

            if not rollback.approved_by:
                raise ValueError("Rollback must be approved before execution")

            rollback.status = RolloutStatus.IN_PROGRESS
            rollback.started_at = datetime.now(timezone.utc)

            # Update the associated rollout
            plan = self._rollouts.get(rollback.rollout_id)
            if plan:
                plan.status = RolloutStatus.ROLLED_BACK

        if self._on_rollback:
            try:
                self._on_rollback(rollback)
            except Exception:
                pass

        return rollback

    def complete_rollback(
        self,
        rollback_id: str,
        agents_rolled_back: int,
        agents_failed: int = 0,
        error: Optional[str] = None,
    ) -> RollbackRequest:
        """Complete a rollback."""
        with self._lock:
            rollback = self._rollbacks.get(rollback_id)
            if not rollback:
                raise ValueError(f"Rollback {rollback_id} not found")

            rollback.completed_at = datetime.now(timezone.utc)
            rollback.agents_rolled_back = agents_rolled_back
            rollback.agents_failed = agents_failed
            rollback.error = error

            if error or agents_failed > agents_rolled_back:
                rollback.status = RolloutStatus.FAILED
            else:
                rollback.status = RolloutStatus.COMPLETED

        # Log event
        event_type = UpdateEventType.ROLLBACK_COMPLETED if rollback.status == RolloutStatus.COMPLETED else UpdateEventType.ROLLBACK_FAILED
        event = UpdateEvent(
            event_type=event_type,
            actor_id="system",
            rollout_id=rollback.rollout_id,
            version=rollback.to_version,
            details={
                "agents_rolled_back": agents_rolled_back,
                "agents_failed": agents_failed,
            },
            result="success" if rollback.status == RolloutStatus.COMPLETED else "failure",
        )
        self._log_event(event)

        return rollback

    def get_rollback(self, rollback_id: str) -> Optional[RollbackRequest]:
        """Get rollback by ID."""
        with self._lock:
            return self._rollbacks.get(rollback_id)

    def list_rollbacks(
        self,
        status: Optional[RolloutStatus] = None,
        limit: int = 100,
    ) -> List[RollbackRequest]:
        """List rollbacks."""
        with self._lock:
            rollbacks = list(self._rollbacks.values())

        if status:
            rollbacks = [r for r in rollbacks if r.status == status]

        rollbacks.sort(key=lambda r: r.requested_at, reverse=True)
        return rollbacks[:limit]

    # =========================================================================
    # Version Pinning
    # =========================================================================

    def pin_version(
        self,
        workspace_id: str,
        version: str,
        pinned_by: str,
        reason: str = "",
        expires_in_days: Optional[int] = None,
        change_window: Optional[ChangeWindow] = None,
    ) -> VersionPin:
        """Pin a workspace to a specific version."""
        if not SEMVER_PATTERN.match(version):
            raise ValueError(f"Invalid semver format: {version}")

        expires_at = None
        if expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

        pin = VersionPin(
            workspace_id=workspace_id,
            pinned_version=version,
            pinned_by=pinned_by,
            reason=reason,
            expires_at=expires_at,
            change_window=change_window,
        )

        with self._lock:
            # Deactivate existing pin if any
            existing_pin_id = self._workspace_pins.get(workspace_id)
            if existing_pin_id:
                existing_pin = self._version_pins.get(existing_pin_id)
                if existing_pin:
                    existing_pin.active = False

            self._version_pins[pin.pin_id] = pin
            self._workspace_pins[workspace_id] = pin.pin_id

        # Log event
        event = UpdateEvent(
            workspace_id=workspace_id,
            event_type=UpdateEventType.VERSION_PINNED,
            actor_id=pinned_by,
            version=version,
            details={
                "reason": reason,
                "expires_at": expires_at.isoformat() if expires_at else None,
                "has_change_window": change_window is not None,
            },
        )
        self._log_event(event)

        return pin

    def unpin_version(
        self,
        workspace_id: str,
        unpinned_by: str,
        reason: str = "",
    ) -> Optional[VersionPin]:
        """Remove version pin for a workspace."""
        with self._lock:
            pin_id = self._workspace_pins.get(workspace_id)
            if not pin_id:
                return None

            pin = self._version_pins.get(pin_id)
            if pin:
                pin.active = False

            del self._workspace_pins[workspace_id]

        # Log event
        event = UpdateEvent(
            workspace_id=workspace_id,
            event_type=UpdateEventType.VERSION_UNPINNED,
            actor_id=unpinned_by,
            version=pin.pinned_version if pin else None,
            details={"reason": reason},
        )
        self._log_event(event)

        return pin

    def get_version_pin(self, workspace_id: str) -> Optional[VersionPin]:
        """Get active version pin for a workspace."""
        with self._lock:
            pin_id = self._workspace_pins.get(workspace_id)
            if not pin_id:
                return None

            pin = self._version_pins.get(pin_id)
            if pin and pin.active and not pin.is_expired():
                return pin

        return None

    def is_update_allowed(
        self,
        workspace_id: str,
        to_version: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if an update is allowed for a workspace.

        Returns: (allowed, reason if blocked)
        """
        pin = self.get_version_pin(workspace_id)
        if not pin:
            return True, None

        # Check version pin
        if pin.pinned_version != to_version:
            return False, f"Workspace is pinned to version {pin.pinned_version}"

        # Check change window
        if pin.change_window:
            if not pin.change_window.is_within_window():
                next_window = pin.change_window.next_window_start()
                return False, f"Outside change window. Next window: {next_window.isoformat()}"

        return True, None

    def set_change_window(
        self,
        workspace_id: str,
        change_window: ChangeWindow,
        set_by: str,
    ) -> Optional[VersionPin]:
        """Set change window for a workspace's version pin."""
        with self._lock:
            pin_id = self._workspace_pins.get(workspace_id)
            if not pin_id:
                return None

            pin = self._version_pins.get(pin_id)
            if pin:
                pin.change_window = change_window

        # Log event
        event = UpdateEvent(
            workspace_id=workspace_id,
            event_type=UpdateEventType.CHANGE_WINDOW_SET,
            actor_id=set_by,
            details={
                "allowed_days": list(change_window.allowed_days),
                "hours": f"{change_window.start_hour}-{change_window.end_hour}",
            },
        )
        self._log_event(event)

        return pin

    def list_version_pins(
        self,
        active_only: bool = True,
        include_expired: bool = False,
    ) -> List[VersionPin]:
        """List version pins."""
        with self._lock:
            pins = list(self._version_pins.values())

        if active_only:
            pins = [p for p in pins if p.active]
        if not include_expired:
            pins = [p for p in pins if not p.is_expired()]

        return pins

    # =========================================================================
    # Statistics and Export
    # =========================================================================

    def get_stats(self, workspace_id: str = "") -> AgentUpdateStats:
        """Get update statistics."""
        with self._lock:
            updates = list(self._updates.values())
            rollouts = list(self._rollouts.values())
            pins = [p for p in self._version_pins.values() if p.active and not p.is_expired()]
            records = list(self._update_records)

        # Filter by workspace if specified
        if workspace_id:
            pins = [p for p in pins if p.workspace_id == workspace_id]
            records = [r for r in records if r.workspace_id == workspace_id]

        # Calculate 30-day metrics
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        recent_records = [r for r in records if r.started_at >= cutoff]

        agents_updated = len([r for r in recent_records if r.result == UpdateResult.SUCCESS])
        agents_failed = len([r for r in recent_records if r.result == UpdateResult.FAILED])
        total = agents_updated + agents_failed
        success_rate = agents_updated / total if total > 0 else 0.0

        # Calculate average rollout duration
        completed_rollouts = [r for r in rollouts if r.status == RolloutStatus.COMPLETED and r.started_at and r.completed_at]
        avg_duration = 0.0
        if completed_rollouts:
            durations = [(r.completed_at - r.started_at).total_seconds() / 3600 for r in completed_rollouts]
            avg_duration = sum(durations) / len(durations)

        return AgentUpdateStats(
            workspace_id=workspace_id,
            total_updates_published=len(updates),
            total_rollouts=len(rollouts),
            rollouts_completed=len([r for r in rollouts if r.status == RolloutStatus.COMPLETED]),
            rollouts_failed=len([r for r in rollouts if r.status == RolloutStatus.FAILED]),
            rollouts_rolled_back=len([r for r in rollouts if r.status == RolloutStatus.ROLLED_BACK]),
            agents_updated_30d=agents_updated,
            agents_failed_30d=agents_failed,
            success_rate_30d=success_rate,
            average_rollout_duration_hours=avg_duration,
            version_pins_active=len(pins),
        )

    def _log_event(self, event: UpdateEvent) -> None:
        """Log an update event."""
        with self._lock:
            self._events.append(event)

        if self._on_event:
            try:
                self._on_event(event)
            except Exception:
                pass

    def get_events(
        self,
        workspace_id: Optional[str] = None,
        event_type: Optional[UpdateEventType] = None,
        rollout_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[UpdateEvent]:
        """Get update events."""
        with self._lock:
            events = list(self._events)

        if workspace_id:
            events = [e for e in events if e.workspace_id == workspace_id]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if rollout_id:
            events = [e for e in events if e.rollout_id == rollout_id]
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    def export_rollout_records(
        self,
        workspace_id: Optional[str] = None,
        include_events: bool = True,
    ) -> Dict[str, Any]:
        """Export rollout records for evidence pack."""
        with self._lock:
            rollouts = list(self._rollouts.values())
            rollbacks = list(self._rollbacks.values())
            pins = list(self._version_pins.values())
            records = list(self._update_records)

        if workspace_id:
            pins = [p for p in pins if p.workspace_id == workspace_id]
            records = [r for r in records if r.workspace_id == workspace_id]

        export_data: Dict[str, Any] = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "workspace_id": workspace_id,
            "rollouts": [r.to_dict() for r in rollouts],
            "rollbacks": [r.to_dict() for r in rollbacks],
            "version_pins": [p.to_dict() for p in pins],
            "update_records": [r.to_dict() for r in records],
        }

        if include_events:
            events = self.get_events(workspace_id=workspace_id, limit=10000)
            export_data["events"] = [e.to_dict() for e in events]

        # Compute hash
        content = json.dumps(export_data, sort_keys=True, default=str)
        export_data["integrity_hash"] = f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"

        return export_data
