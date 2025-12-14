# -*- coding: utf-8 -*-
"""
Version Pinning Manager - Enterprise version control and change windows.

CLOUD ZONE ONLY.

Provides enterprise version management:
- Version pinning per workspace/agent
- Change windows (allowed update times)
- Min/max supported version constraints
- Schema version compatibility

Design Doc Reference:
    - Phase 9: Enterprise/on-prem pack
    - Design Doc 15.2: Agent updates - Version pinning + change windows
    - Design Doc 10: Protocol - schema_versioning + min_supported/max_supported
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, Final, List, Optional, Set, Tuple
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class PinType(Enum):
    """Type of version pin."""
    EXACT = "exact"  # Pin to exact version
    MAJOR = "major"  # Pin to major version (e.g., 1.x.x)
    MINOR = "minor"  # Pin to minor version (e.g., 1.2.x)
    RANGE = "range"  # Pin to version range
    LATEST = "latest"  # Always use latest


class PinScope(Enum):
    """Scope of version pin."""
    GLOBAL = "global"  # All agents
    ORGANIZATION = "organization"  # Organization-wide
    WORKSPACE = "workspace"  # Workspace-specific
    AGENT = "agent"  # Single agent


class WindowType(Enum):
    """Type of change window."""
    MAINTENANCE = "maintenance"  # Regular maintenance window
    EMERGENCY = "emergency"  # Emergency updates (bypasses restrictions)
    BLACKOUT = "blackout"  # No changes allowed


@dataclass
class VersionConstraint:
    """Version constraint definition."""
    id: UUID = field(default_factory=uuid4)
    name: str = ""

    # Constraint type
    pin_type: PinType = PinType.RANGE
    pin_scope: PinScope = PinScope.WORKSPACE

    # Target
    workspace_id: Optional[UUID] = None
    organization_id: Optional[UUID] = None
    agent_id: Optional[UUID] = None

    # Version specification
    version: str = ""  # For EXACT
    min_version: str = ""  # Minimum allowed
    max_version: str = ""  # Maximum allowed
    excluded_versions: List[str] = field(default_factory=list)

    # Schema compatibility
    min_schema_version: str = "1.0"
    max_schema_version: str = ""

    # Behavior
    allow_downgrade: bool = False
    auto_update: bool = False
    require_approval: bool = True

    # Metadata
    reason: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    expires_at: Optional[datetime] = None
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "pin_type": self.pin_type.value,
            "pin_scope": self.pin_scope.value,
            "workspace_id": str(self.workspace_id) if self.workspace_id else None,
            "organization_id": str(self.organization_id) if self.organization_id else None,
            "agent_id": str(self.agent_id) if self.agent_id else None,
            "version": self.version,
            "min_version": self.min_version,
            "max_version": self.max_version,
            "excluded_versions": self.excluded_versions,
            "min_schema_version": self.min_schema_version,
            "max_schema_version": self.max_schema_version,
            "allow_downgrade": self.allow_downgrade,
            "auto_update": self.auto_update,
            "require_approval": self.require_approval,
            "reason": self.reason,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
        }


@dataclass
class ChangeWindow:
    """Change window definition."""
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    window_type: WindowType = WindowType.MAINTENANCE

    # Scope
    workspace_id: Optional[UUID] = None
    organization_id: Optional[UUID] = None

    # Schedule (recurring)
    is_recurring: bool = True
    days_of_week: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri
    start_time: time = field(default_factory=lambda: time(9, 0))  # 09:00
    end_time: time = field(default_factory=lambda: time(17, 0))  # 17:00
    timezone_name: str = "UTC"

    # One-time window
    start_datetime: Optional[datetime] = None
    end_datetime: Optional[datetime] = None

    # Behavior
    allow_emergency: bool = True  # Allow emergency updates outside window
    require_approval: bool = True
    notify_before_minutes: int = 30

    # Metadata
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "window_type": self.window_type.value,
            "workspace_id": str(self.workspace_id) if self.workspace_id else None,
            "organization_id": str(self.organization_id) if self.organization_id else None,
            "is_recurring": self.is_recurring,
            "days_of_week": self.days_of_week,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "timezone_name": self.timezone_name,
            "start_datetime": self.start_datetime.isoformat() if self.start_datetime else None,
            "end_datetime": self.end_datetime.isoformat() if self.end_datetime else None,
            "allow_emergency": self.allow_emergency,
            "require_approval": self.require_approval,
            "description": self.description,
            "is_active": self.is_active,
        }


@dataclass
class VersionPinConfig:
    """Configuration for version pinning manager."""
    # Global constraints
    global_min_version: str = "0.9.0"
    global_max_version: str = ""  # Empty = no max
    global_min_schema_version: str = "1.0"
    global_max_schema_version: str = ""

    # Default behavior
    default_pin_type: PinType = PinType.RANGE
    default_allow_downgrade: bool = False
    default_auto_update: bool = False
    default_require_approval: bool = True

    # Change windows
    enable_change_windows: bool = False
    default_window_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    default_window_start: time = field(default_factory=lambda: time(9, 0))
    default_window_end: time = field(default_factory=lambda: time(17, 0))
    default_timezone: str = "UTC"

    # Notifications
    notify_on_pin_change: bool = True
    notify_on_version_block: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "global_min_version": self.global_min_version,
            "global_max_version": self.global_max_version,
            "global_min_schema_version": self.global_min_schema_version,
            "global_max_schema_version": self.global_max_schema_version,
            "default_pin_type": self.default_pin_type.value,
            "default_allow_downgrade": self.default_allow_downgrade,
            "default_auto_update": self.default_auto_update,
            "default_require_approval": self.default_require_approval,
            "enable_change_windows": self.enable_change_windows,
        }


@dataclass
class VersionCheckResult:
    """Result of version compatibility check."""
    is_allowed: bool = False
    reason: str = ""
    constraint_id: Optional[UUID] = None
    constraint_name: str = ""

    # Details
    current_version: str = ""
    target_version: str = ""
    min_allowed: str = ""
    max_allowed: str = ""

    # Window info
    in_change_window: bool = True
    next_window_start: Optional[datetime] = None

    # Recommendations
    recommended_version: Optional[str] = None
    requires_approval: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_allowed": self.is_allowed,
            "reason": self.reason,
            "constraint_id": str(self.constraint_id) if self.constraint_id else None,
            "constraint_name": self.constraint_name,
            "current_version": self.current_version,
            "target_version": self.target_version,
            "min_allowed": self.min_allowed,
            "max_allowed": self.max_allowed,
            "in_change_window": self.in_change_window,
            "next_window_start": self.next_window_start.isoformat() if self.next_window_start else None,
            "recommended_version": self.recommended_version,
            "requires_approval": self.requires_approval,
        }


class VersionPinningManager:
    """
    Version Pinning Manager for enterprise version control.

    Features:
    - Version constraints per workspace/agent
    - Change window enforcement
    - Schema version compatibility
    - Upgrade/downgrade control

    Usage:
        config = VersionPinConfig(enable_change_windows=True)
        manager = VersionPinningManager(config)

        # Create constraint
        constraint = manager.create_constraint(
            name="Production Pin",
            pin_type=PinType.MINOR,
            workspace_id=workspace_id,
            min_version="1.0.0",
            max_version="1.2.99",
        )

        # Check version
        result = manager.check_version(
            agent_id=agent_id,
            current_version="1.0.0",
            target_version="1.1.0",
        )
        if result.is_allowed:
            # Proceed with update
            pass
    """

    def __init__(
        self,
        config: Optional[VersionPinConfig] = None,
        db_session: Optional[Any] = None,
        on_version_blocked: Optional[Callable[[UUID, str, str], None]] = None,
    ):
        """
        Initialize version pinning manager.

        Args:
            config: Configuration
            db_session: Database session
            on_version_blocked: Callback when version is blocked
        """
        self.config = config or VersionPinConfig()
        self._db = db_session
        self._on_version_blocked = on_version_blocked

        # State
        self._constraints: Dict[UUID, VersionConstraint] = {}
        self._windows: Dict[UUID, ChangeWindow] = {}

        # Cache
        self._agent_constraints: Dict[UUID, List[VersionConstraint]] = {}

    def create_constraint(
        self,
        name: str,
        pin_type: PinType = PinType.RANGE,
        pin_scope: PinScope = PinScope.WORKSPACE,
        workspace_id: Optional[UUID] = None,
        organization_id: Optional[UUID] = None,
        agent_id: Optional[UUID] = None,
        version: str = "",
        min_version: str = "",
        max_version: str = "",
        excluded_versions: Optional[List[str]] = None,
        min_schema_version: str = "",
        max_schema_version: str = "",
        allow_downgrade: bool = False,
        auto_update: bool = False,
        require_approval: bool = True,
        reason: str = "",
        expires_at: Optional[datetime] = None,
        created_by: str = "system",
    ) -> VersionConstraint:
        """
        Create a version constraint.

        Args:
            name: Constraint name
            pin_type: Type of pin
            pin_scope: Scope of constraint
            workspace_id: Target workspace
            organization_id: Target organization
            agent_id: Target agent
            version: Exact version (for EXACT pin)
            min_version: Minimum version
            max_version: Maximum version
            excluded_versions: Versions to exclude
            min_schema_version: Minimum schema version
            max_schema_version: Maximum schema version
            allow_downgrade: Allow downgrades
            auto_update: Auto-update within constraints
            require_approval: Require approval for updates
            reason: Reason for constraint
            expires_at: Constraint expiration
            created_by: Creator

        Returns:
            Created constraint
        """
        constraint = VersionConstraint(
            name=name,
            pin_type=pin_type,
            pin_scope=pin_scope,
            workspace_id=workspace_id,
            organization_id=organization_id,
            agent_id=agent_id,
            version=version,
            min_version=min_version or self.config.global_min_version,
            max_version=max_version or self.config.global_max_version,
            excluded_versions=excluded_versions or [],
            min_schema_version=min_schema_version or self.config.global_min_schema_version,
            max_schema_version=max_schema_version or self.config.global_max_schema_version,
            allow_downgrade=allow_downgrade,
            auto_update=auto_update,
            require_approval=require_approval,
            reason=reason,
            expires_at=expires_at,
            created_by=created_by,
        )

        self._constraints[constraint.id] = constraint

        # Invalidate cache
        self._agent_constraints.clear()

        logger.info(f"Created version constraint: {name} ({constraint.id})")
        return constraint

    def create_change_window(
        self,
        name: str,
        window_type: WindowType = WindowType.MAINTENANCE,
        workspace_id: Optional[UUID] = None,
        organization_id: Optional[UUID] = None,
        is_recurring: bool = True,
        days_of_week: Optional[List[int]] = None,
        start_time: Optional[time] = None,
        end_time: Optional[time] = None,
        timezone_name: str = "UTC",
        start_datetime: Optional[datetime] = None,
        end_datetime: Optional[datetime] = None,
        allow_emergency: bool = True,
        require_approval: bool = True,
        description: str = "",
        created_by: str = "system",
    ) -> ChangeWindow:
        """
        Create a change window.

        Args:
            name: Window name
            window_type: Type of window
            workspace_id: Target workspace
            organization_id: Target organization
            is_recurring: Is recurring window
            days_of_week: Days when window is active (0=Mon)
            start_time: Daily start time
            end_time: Daily end time
            timezone_name: Timezone
            start_datetime: One-time window start
            end_datetime: One-time window end
            allow_emergency: Allow emergency updates
            require_approval: Require approval
            description: Description
            created_by: Creator

        Returns:
            Created window
        """
        window = ChangeWindow(
            name=name,
            window_type=window_type,
            workspace_id=workspace_id,
            organization_id=organization_id,
            is_recurring=is_recurring,
            days_of_week=days_of_week or self.config.default_window_days,
            start_time=start_time or self.config.default_window_start,
            end_time=end_time or self.config.default_window_end,
            timezone_name=timezone_name or self.config.default_timezone,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            allow_emergency=allow_emergency,
            require_approval=require_approval,
            description=description,
            created_by=created_by,
        )

        self._windows[window.id] = window
        logger.info(f"Created change window: {name} ({window.id})")

        return window

    def check_version(
        self,
        agent_id: UUID,
        current_version: str,
        target_version: str,
        workspace_id: Optional[UUID] = None,
        organization_id: Optional[UUID] = None,
        schema_version: str = "",
        is_emergency: bool = False,
    ) -> VersionCheckResult:
        """
        Check if version update is allowed.

        Args:
            agent_id: Agent ID
            current_version: Current agent version
            target_version: Target version
            workspace_id: Agent's workspace
            organization_id: Agent's organization
            schema_version: Protocol schema version
            is_emergency: Is emergency update

        Returns:
            Check result
        """
        result = VersionCheckResult(
            current_version=current_version,
            target_version=target_version,
        )

        # Get applicable constraints
        constraints = self._get_applicable_constraints(
            agent_id, workspace_id, organization_id
        )

        # Check each constraint
        for constraint in constraints:
            if not constraint.is_active:
                continue

            # Check expiration
            if constraint.expires_at and datetime.utcnow() > constraint.expires_at:
                continue

            # Check version against constraint
            allowed, reason = self._check_constraint(
                constraint, current_version, target_version
            )

            if not allowed:
                result.is_allowed = False
                result.reason = reason
                result.constraint_id = constraint.id
                result.constraint_name = constraint.name
                result.min_allowed = constraint.min_version
                result.max_allowed = constraint.max_version
                result.requires_approval = constraint.require_approval

                # Notify
                if self._on_version_blocked:
                    try:
                        self._on_version_blocked(
                            agent_id, target_version, reason
                        )
                    except Exception as e:
                        logger.error(f"Version blocked callback error: {e}")

                return result

            # Track constraint info
            result.min_allowed = constraint.min_version
            result.max_allowed = constraint.max_version
            result.requires_approval = constraint.require_approval

        # Check change window
        if self.config.enable_change_windows and not is_emergency:
            in_window, next_start = self._check_change_window(
                workspace_id, organization_id
            )
            result.in_change_window = in_window
            result.next_window_start = next_start

            if not in_window:
                result.is_allowed = False
                result.reason = f"Outside change window. Next window starts at {next_start}"
                return result

        # All checks passed
        result.is_allowed = True
        result.reason = "Version allowed"
        return result

    def check_schema_version(
        self,
        schema_version: str,
        workspace_id: Optional[UUID] = None,
    ) -> Tuple[bool, str]:
        """
        Check if schema version is compatible.

        Args:
            schema_version: Schema version to check
            workspace_id: Workspace ID

        Returns:
            (is_compatible, reason)
        """
        from packaging import version as pkg_version

        try:
            schema_v = pkg_version.parse(schema_version)

            # Check global constraints
            if self.config.global_min_schema_version:
                min_v = pkg_version.parse(self.config.global_min_schema_version)
                if schema_v < min_v:
                    return False, f"Schema version {schema_version} below minimum {self.config.global_min_schema_version}"

            if self.config.global_max_schema_version:
                max_v = pkg_version.parse(self.config.global_max_schema_version)
                if schema_v > max_v:
                    return False, f"Schema version {schema_version} above maximum {self.config.global_max_schema_version}"

            # Check workspace constraints
            if workspace_id:
                for constraint in self._constraints.values():
                    if constraint.workspace_id == workspace_id and constraint.is_active:
                        if constraint.min_schema_version:
                            min_v = pkg_version.parse(constraint.min_schema_version)
                            if schema_v < min_v:
                                return False, f"Schema version below workspace minimum"

                        if constraint.max_schema_version:
                            max_v = pkg_version.parse(constraint.max_schema_version)
                            if schema_v > max_v:
                                return False, f"Schema version above workspace maximum"

            return True, "Schema version compatible"

        except Exception as e:
            return False, f"Version parsing error: {e}"

    def get_allowed_versions(
        self,
        agent_id: UUID,
        workspace_id: Optional[UUID] = None,
        organization_id: Optional[UUID] = None,
    ) -> Tuple[str, str, List[str]]:
        """
        Get allowed version range for an agent.

        Args:
            agent_id: Agent ID
            workspace_id: Workspace ID
            organization_id: Organization ID

        Returns:
            (min_version, max_version, excluded_versions)
        """
        constraints = self._get_applicable_constraints(
            agent_id, workspace_id, organization_id
        )

        min_version = self.config.global_min_version
        max_version = self.config.global_max_version
        excluded: Set[str] = set()

        for constraint in constraints:
            if not constraint.is_active:
                continue

            # Most restrictive min
            if constraint.min_version:
                if not min_version or self._compare_versions(constraint.min_version, min_version) > 0:
                    min_version = constraint.min_version

            # Most restrictive max
            if constraint.max_version:
                if not max_version or self._compare_versions(constraint.max_version, max_version) < 0:
                    max_version = constraint.max_version

            # Union of excluded
            excluded.update(constraint.excluded_versions)

        return min_version, max_version, list(excluded)

    def get_constraint(self, constraint_id: UUID) -> Optional[VersionConstraint]:
        """Get constraint by ID."""
        return self._constraints.get(constraint_id)

    def get_window(self, window_id: UUID) -> Optional[ChangeWindow]:
        """Get change window by ID."""
        return self._windows.get(window_id)

    def update_constraint(
        self,
        constraint_id: UUID,
        **updates: Any,
    ) -> Optional[VersionConstraint]:
        """
        Update a constraint.

        Args:
            constraint_id: Constraint ID
            **updates: Fields to update

        Returns:
            Updated constraint or None
        """
        constraint = self._constraints.get(constraint_id)
        if not constraint:
            return None

        for key, value in updates.items():
            if hasattr(constraint, key):
                setattr(constraint, key, value)

        # Invalidate cache
        self._agent_constraints.clear()

        logger.info(f"Updated constraint: {constraint.name}")
        return constraint

    def delete_constraint(self, constraint_id: UUID) -> bool:
        """Delete a constraint."""
        if constraint_id in self._constraints:
            constraint = self._constraints.pop(constraint_id)
            self._agent_constraints.clear()
            logger.info(f"Deleted constraint: {constraint.name}")
            return True
        return False

    def delete_window(self, window_id: UUID) -> bool:
        """Delete a change window."""
        if window_id in self._windows:
            window = self._windows.pop(window_id)
            logger.info(f"Deleted change window: {window.name}")
            return True
        return False

    def list_constraints(
        self,
        workspace_id: Optional[UUID] = None,
        active_only: bool = True,
    ) -> List[VersionConstraint]:
        """List constraints."""
        constraints = list(self._constraints.values())

        if workspace_id:
            constraints = [
                c for c in constraints
                if c.workspace_id == workspace_id or c.pin_scope == PinScope.GLOBAL
            ]

        if active_only:
            constraints = [c for c in constraints if c.is_active]

        return constraints

    def list_windows(
        self,
        workspace_id: Optional[UUID] = None,
        active_only: bool = True,
    ) -> List[ChangeWindow]:
        """List change windows."""
        windows = list(self._windows.values())

        if workspace_id:
            windows = [
                w for w in windows
                if w.workspace_id == workspace_id or not w.workspace_id
            ]

        if active_only:
            windows = [w for w in windows if w.is_active]

        return windows

    def is_in_change_window(
        self,
        workspace_id: Optional[UUID] = None,
        organization_id: Optional[UUID] = None,
    ) -> bool:
        """Check if currently in a change window."""
        in_window, _ = self._check_change_window(workspace_id, organization_id)
        return in_window

    def get_next_change_window(
        self,
        workspace_id: Optional[UUID] = None,
    ) -> Optional[datetime]:
        """Get next change window start time."""
        _, next_start = self._check_change_window(workspace_id, None)
        return next_start

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            "total_constraints": len(self._constraints),
            "active_constraints": sum(
                1 for c in self._constraints.values() if c.is_active
            ),
            "total_windows": len(self._windows),
            "active_windows": sum(
                1 for w in self._windows.values() if w.is_active
            ),
            "config": self.config.to_dict(),
        }

    # ===== Private Methods =====

    def _get_applicable_constraints(
        self,
        agent_id: UUID,
        workspace_id: Optional[UUID],
        organization_id: Optional[UUID],
    ) -> List[VersionConstraint]:
        """Get constraints applicable to an agent."""
        # Check cache
        if agent_id in self._agent_constraints:
            return self._agent_constraints[agent_id]

        applicable = []

        for constraint in self._constraints.values():
            if not constraint.is_active:
                continue

            # Check scope
            if constraint.pin_scope == PinScope.GLOBAL:
                applicable.append(constraint)
            elif constraint.pin_scope == PinScope.ORGANIZATION:
                if organization_id and constraint.organization_id == organization_id:
                    applicable.append(constraint)
            elif constraint.pin_scope == PinScope.WORKSPACE:
                if workspace_id and constraint.workspace_id == workspace_id:
                    applicable.append(constraint)
            elif constraint.pin_scope == PinScope.AGENT:
                if constraint.agent_id == agent_id:
                    applicable.append(constraint)

        # Sort by specificity (agent > workspace > org > global)
        scope_order = {
            PinScope.AGENT: 0,
            PinScope.WORKSPACE: 1,
            PinScope.ORGANIZATION: 2,
            PinScope.GLOBAL: 3,
        }
        applicable.sort(key=lambda c: scope_order[c.pin_scope])

        # Cache
        self._agent_constraints[agent_id] = applicable
        return applicable

    def _check_constraint(
        self,
        constraint: VersionConstraint,
        current: str,
        target: str,
    ) -> Tuple[bool, str]:
        """Check version against constraint."""
        from packaging import version as pkg_version

        try:
            current_v = pkg_version.parse(current)
            target_v = pkg_version.parse(target)

            # Check downgrade
            if target_v < current_v and not constraint.allow_downgrade:
                return False, "Downgrade not allowed by constraint"

            # Check exact pin
            if constraint.pin_type == PinType.EXACT:
                if constraint.version and target != constraint.version:
                    return False, f"Version pinned to {constraint.version}"

            # Check major pin
            elif constraint.pin_type == PinType.MAJOR:
                if constraint.version:
                    pin_v = pkg_version.parse(constraint.version)
                    if target_v.major != pin_v.major:
                        return False, f"Major version pinned to {pin_v.major}.x.x"

            # Check minor pin
            elif constraint.pin_type == PinType.MINOR:
                if constraint.version:
                    pin_v = pkg_version.parse(constraint.version)
                    if target_v.major != pin_v.major or target_v.minor != pin_v.minor:
                        return False, f"Minor version pinned to {pin_v.major}.{pin_v.minor}.x"

            # Check range
            if constraint.min_version:
                min_v = pkg_version.parse(constraint.min_version)
                if target_v < min_v:
                    return False, f"Version below minimum {constraint.min_version}"

            if constraint.max_version:
                max_v = pkg_version.parse(constraint.max_version)
                if target_v > max_v:
                    return False, f"Version above maximum {constraint.max_version}"

            # Check excluded
            if target in constraint.excluded_versions:
                return False, f"Version {target} is excluded"

            return True, "Version allowed"

        except Exception as e:
            return False, f"Version check error: {e}"

    def _check_change_window(
        self,
        workspace_id: Optional[UUID],
        organization_id: Optional[UUID],
    ) -> Tuple[bool, Optional[datetime]]:
        """Check if currently in a change window."""
        if not self.config.enable_change_windows:
            return True, None

        now = datetime.now(timezone.utc)
        next_window_start: Optional[datetime] = None

        # Get applicable windows
        applicable_windows = []
        for window in self._windows.values():
            if not window.is_active:
                continue

            # Check scope
            if window.workspace_id and window.workspace_id != workspace_id:
                continue
            if window.organization_id and window.organization_id != organization_id:
                continue

            applicable_windows.append(window)

        if not applicable_windows:
            # No windows defined = always allowed
            return True, None

        # Check each window
        for window in applicable_windows:
            if window.window_type == WindowType.BLACKOUT:
                # Check if in blackout
                if window.is_recurring:
                    if self._is_in_recurring_window(window, now):
                        return False, self._get_window_end(window, now)
                elif window.start_datetime and window.end_datetime:
                    if window.start_datetime <= now <= window.end_datetime:
                        return False, window.end_datetime

            elif window.window_type == WindowType.MAINTENANCE:
                # Check if in maintenance window
                if window.is_recurring:
                    if self._is_in_recurring_window(window, now):
                        return True, None
                    # Track next window
                    next_start = self._get_next_window_start(window, now)
                    if next_start:
                        if not next_window_start or next_start < next_window_start:
                            next_window_start = next_start
                elif window.start_datetime and window.end_datetime:
                    if window.start_datetime <= now <= window.end_datetime:
                        return True, None

        # Not in any maintenance window
        return False, next_window_start

    def _is_in_recurring_window(self, window: ChangeWindow, now: datetime) -> bool:
        """Check if time is in recurring window."""
        # Check day of week
        if now.weekday() not in window.days_of_week:
            return False

        # Check time
        current_time = now.time()
        return window.start_time <= current_time < window.end_time

    def _get_window_end(self, window: ChangeWindow, now: datetime) -> datetime:
        """Get end time of current window."""
        return datetime.combine(now.date(), window.end_time, tzinfo=timezone.utc)

    def _get_next_window_start(
        self,
        window: ChangeWindow,
        now: datetime,
    ) -> Optional[datetime]:
        """Get next window start time."""
        current_date = now.date()

        # Check today
        if now.weekday() in window.days_of_week:
            if now.time() < window.start_time:
                return datetime.combine(
                    current_date, window.start_time, tzinfo=timezone.utc
                )

        # Find next valid day
        for i in range(1, 8):
            next_date = current_date + timedelta(days=i)
            if next_date.weekday() in window.days_of_week:
                return datetime.combine(
                    next_date, window.start_time, tzinfo=timezone.utc
                )

        return None

    def _compare_versions(self, v1: str, v2: str) -> int:
        """Compare two versions. Returns -1, 0, or 1."""
        from packaging import version as pkg_version

        try:
            ver1 = pkg_version.parse(v1)
            ver2 = pkg_version.parse(v2)

            if ver1 < ver2:
                return -1
            elif ver1 > ver2:
                return 1
            return 0
        except Exception:
            return 0
