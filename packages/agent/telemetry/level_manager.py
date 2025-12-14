# -*- coding: utf-8 -*-
"""
Telemetry Level Manager.

CCEA Phase 8 Implementation.

This module manages telemetry verbosity levels and enforces
defaults based on deployment mode (retail vs enterprise).

Key Features:
    - AGGREGATED as default for retail users
    - DETAILED_NON_SENSITIVE as opt-in option
    - RAW_ORDER_EVENTS only for enterprise with explicit opt-in
    - Level transition controls and validation
    - Integration with DLP for level-appropriate filtering

Design Doc Reference:
    - Phase 8 (13.1): "Telemetry levels and defaults"
    - "AGGREGATED default (retail)"
    - "RAW_ORDER_EVENTS only opt-in and disabled by default"

SECURITY:
    - RAW_ORDER_EVENTS requires enterprise tier AND explicit opt-in
    - Level changes are logged for audit
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, Final, List, Optional, Set
from uuid import uuid4


# ============================================================================
# Enums and Constants
# ============================================================================

class TelemetryLevel(Enum):
    """
    Telemetry detail level.

    From Design Doc 13.1:
    - AGGREGATED: Default for retail, minimal data
    - DETAILED_NON_SENSITIVE: More detail, still safe
    - RAW_ORDER_EVENTS: Enterprise only, explicit opt-in required
    """
    AGGREGATED = "aggregated"
    DETAILED_NON_SENSITIVE = "detailed_non_sensitive"
    RAW_ORDER_EVENTS = "raw_order_events"


class TelemetryMode(Enum):
    """Deployment mode affecting telemetry defaults."""
    RETAIL = auto()      # Consumer/individual use
    PROFESSIONAL = auto()  # Professional traders
    ENTERPRISE = auto()  # Enterprise/institutional


# Default levels per mode
DEFAULT_LEVELS: Final[Dict[TelemetryMode, TelemetryLevel]] = {
    TelemetryMode.RETAIL: TelemetryLevel.AGGREGATED,
    TelemetryMode.PROFESSIONAL: TelemetryLevel.AGGREGATED,
    TelemetryMode.ENTERPRISE: TelemetryLevel.DETAILED_NON_SENSITIVE,
}

# Maximum allowed levels per mode
MAX_LEVELS: Final[Dict[TelemetryMode, TelemetryLevel]] = {
    TelemetryMode.RETAIL: TelemetryLevel.DETAILED_NON_SENSITIVE,
    TelemetryMode.PROFESSIONAL: TelemetryLevel.DETAILED_NON_SENSITIVE,
    TelemetryMode.ENTERPRISE: TelemetryLevel.RAW_ORDER_EVENTS,
}

# Level hierarchy (higher index = more verbose)
LEVEL_HIERARCHY: Final[List[TelemetryLevel]] = [
    TelemetryLevel.AGGREGATED,
    TelemetryLevel.DETAILED_NON_SENSITIVE,
    TelemetryLevel.RAW_ORDER_EVENTS,
]


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TelemetryLevelConfig:
    """
    Configuration for telemetry level management.

    Attributes:
        mode: Deployment mode (retail/professional/enterprise)
        current_level: Currently active telemetry level
        max_allowed_level: Maximum level allowed for this deployment
        raw_events_opt_in: Explicit opt-in for RAW_ORDER_EVENTS
        level_change_requires_approval: Require approval for level changes
    """
    mode: TelemetryMode = TelemetryMode.RETAIL
    current_level: TelemetryLevel = TelemetryLevel.AGGREGATED
    max_allowed_level: TelemetryLevel = TelemetryLevel.DETAILED_NON_SENSITIVE
    raw_events_opt_in: bool = False
    level_change_requires_approval: bool = False

    def __post_init__(self):
        """Validate configuration after init."""
        # Enforce mode-based max level
        mode_max = MAX_LEVELS.get(self.mode, TelemetryLevel.DETAILED_NON_SENSITIVE)
        mode_max_idx = LEVEL_HIERARCHY.index(mode_max)
        config_max_idx = LEVEL_HIERARCHY.index(self.max_allowed_level)

        if config_max_idx > mode_max_idx:
            self.max_allowed_level = mode_max

        # RAW_ORDER_EVENTS requires both enterprise AND opt-in
        if self.current_level == TelemetryLevel.RAW_ORDER_EVENTS:
            if self.mode != TelemetryMode.ENTERPRISE or not self.raw_events_opt_in:
                self.current_level = TelemetryLevel.DETAILED_NON_SENSITIVE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mode": self.mode.name,
            "current_level": self.current_level.value,
            "max_allowed_level": self.max_allowed_level.value,
            "raw_events_opt_in": self.raw_events_opt_in,
            "level_change_requires_approval": self.level_change_requires_approval,
        }


@dataclass
class LevelChangeRequest:
    """Request to change telemetry level."""
    id: str = field(default_factory=lambda: str(uuid4()))
    requested_level: TelemetryLevel = TelemetryLevel.AGGREGATED
    requested_by: str = ""
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    approved: Optional[bool] = None
    approved_by: Optional[str] = None
    approval_timestamp: Optional[datetime] = None


@dataclass
class LevelChangeResult:
    """Result of level change operation."""
    success: bool
    previous_level: TelemetryLevel
    new_level: TelemetryLevel
    message: str
    requires_approval: bool = False
    request_id: Optional[str] = None


@dataclass
class TelemetryFilter:
    """
    Filter configuration for telemetry data based on level.

    Defines what data is included at each level.
    """
    level: TelemetryLevel

    # Fields to include at this level
    include_fields: Set[str] = field(default_factory=set)

    # Fields to exclude at this level
    exclude_fields: Set[str] = field(default_factory=set)

    # Whether to include individual order events
    include_order_events: bool = False

    # Whether to include position details
    include_position_details: bool = False

    # Whether to aggregate metrics
    aggregate_metrics: bool = True

    # Aggregation window in seconds
    aggregation_window: int = 60


# Default filters per level
LEVEL_FILTERS: Final[Dict[TelemetryLevel, TelemetryFilter]] = {
    TelemetryLevel.AGGREGATED: TelemetryFilter(
        level=TelemetryLevel.AGGREGATED,
        include_fields={"status", "pnl_summary", "error_count", "health"},
        exclude_fields={"orders", "positions", "trades", "fills"},
        include_order_events=False,
        include_position_details=False,
        aggregate_metrics=True,
        aggregation_window=60,
    ),
    TelemetryLevel.DETAILED_NON_SENSITIVE: TelemetryFilter(
        level=TelemetryLevel.DETAILED_NON_SENSITIVE,
        include_fields={"status", "pnl_summary", "error_count", "health", "metrics", "events", "positions"},
        exclude_fields={"raw_orders", "raw_fills", "account_balance"},
        include_order_events=False,
        include_position_details=True,
        aggregate_metrics=True,
        aggregation_window=30,
    ),
    TelemetryLevel.RAW_ORDER_EVENTS: TelemetryFilter(
        level=TelemetryLevel.RAW_ORDER_EVENTS,
        include_fields={"*"},  # All fields
        exclude_fields={"credentials", "api_keys", "secrets"},  # Still exclude secrets
        include_order_events=True,
        include_position_details=True,
        aggregate_metrics=False,
        aggregation_window=0,
    ),
}


class TelemetryLevelManager:
    """
    Manages telemetry verbosity levels.

    This manager controls what data is included in telemetry based
    on the deployment mode and configured level.

    Usage:
        manager = TelemetryLevelManager(TelemetryLevelConfig(
            mode=TelemetryMode.RETAIL,
        ))

        # Check current level
        print(f"Level: {manager.current_level}")

        # Filter telemetry data
        filtered_data = manager.filter_telemetry(raw_data)

        # Request level change
        result = manager.request_level_change(
            TelemetryLevel.DETAILED_NON_SENSITIVE,
            reason="Debugging issue"
        )

    SECURITY:
        - RAW_ORDER_EVENTS requires enterprise + opt-in
        - Level changes are logged for audit
    """

    def __init__(
        self,
        config: Optional[TelemetryLevelConfig] = None,
        on_level_change: Optional[Callable[[LevelChangeResult], None]] = None,
    ):
        """
        Initialize level manager.

        Args:
            config: Level configuration
            on_level_change: Callback for level changes
        """
        self.config = config or TelemetryLevelConfig()
        self._on_level_change = on_level_change
        self._pending_requests: Dict[str, LevelChangeRequest] = {}
        self._change_history: List[LevelChangeResult] = []
        self._lock = threading.Lock()

        # Apply initial configuration
        self._apply_mode_defaults()

    def _apply_mode_defaults(self) -> None:
        """Apply defaults based on deployment mode."""
        # Set default level if not explicitly configured
        default = DEFAULT_LEVELS.get(self.config.mode, TelemetryLevel.AGGREGATED)
        max_allowed = MAX_LEVELS.get(self.config.mode, TelemetryLevel.DETAILED_NON_SENSITIVE)

        # Ensure current level doesn't exceed max
        current_idx = LEVEL_HIERARCHY.index(self.config.current_level)
        max_idx = LEVEL_HIERARCHY.index(max_allowed)

        if current_idx > max_idx:
            self.config.current_level = max_allowed

        self.config.max_allowed_level = max_allowed

    @property
    def current_level(self) -> TelemetryLevel:
        """Get current telemetry level."""
        return self.config.current_level

    @property
    def mode(self) -> TelemetryMode:
        """Get deployment mode."""
        return self.config.mode

    def get_filter(self) -> TelemetryFilter:
        """Get filter configuration for current level."""
        return LEVEL_FILTERS.get(
            self.config.current_level,
            LEVEL_FILTERS[TelemetryLevel.AGGREGATED]
        )

    def can_change_to(self, level: TelemetryLevel) -> tuple[bool, str]:
        """
        Check if level change is allowed.

        Args:
            level: Target level

        Returns:
            Tuple of (allowed, reason)
        """
        # RAW_ORDER_EVENTS has special requirements
        if level == TelemetryLevel.RAW_ORDER_EVENTS:
            if self.config.mode != TelemetryMode.ENTERPRISE:
                return False, "RAW_ORDER_EVENTS requires Enterprise tier"
            if not self.config.raw_events_opt_in:
                return False, "RAW_ORDER_EVENTS requires explicit opt-in"

        # Check against max allowed
        target_idx = LEVEL_HIERARCHY.index(level)
        max_idx = LEVEL_HIERARCHY.index(self.config.max_allowed_level)

        if target_idx > max_idx:
            return False, f"Level exceeds maximum allowed ({self.config.max_allowed_level.value})"

        return True, "Allowed"

    def request_level_change(
        self,
        level: TelemetryLevel,
        requested_by: str = "system",
        reason: str = "",
    ) -> LevelChangeResult:
        """
        Request a change to telemetry level.

        Args:
            level: Target level
            requested_by: Who is requesting
            reason: Reason for change

        Returns:
            LevelChangeResult
        """
        with self._lock:
            previous = self.config.current_level

            # Check if change is allowed
            allowed, message = self.can_change_to(level)
            if not allowed:
                return LevelChangeResult(
                    success=False,
                    previous_level=previous,
                    new_level=previous,
                    message=message,
                )

            # Check if approval is required
            if self.config.level_change_requires_approval:
                request = LevelChangeRequest(
                    requested_level=level,
                    requested_by=requested_by,
                    reason=reason,
                )
                self._pending_requests[request.id] = request

                return LevelChangeResult(
                    success=False,
                    previous_level=previous,
                    new_level=previous,
                    message="Level change requires approval",
                    requires_approval=True,
                    request_id=request.id,
                )

            # Apply change directly
            self.config.current_level = level
            result = LevelChangeResult(
                success=True,
                previous_level=previous,
                new_level=level,
                message=f"Level changed from {previous.value} to {level.value}",
            )

            self._change_history.append(result)

            # Notify callback
            if self._on_level_change:
                self._on_level_change(result)

            return result

    def approve_level_change(
        self,
        request_id: str,
        approved_by: str,
        approve: bool = True,
    ) -> LevelChangeResult:
        """
        Approve or reject a pending level change request.

        Args:
            request_id: Request to approve
            approved_by: Who is approving
            approve: Whether to approve

        Returns:
            LevelChangeResult
        """
        with self._lock:
            if request_id not in self._pending_requests:
                return LevelChangeResult(
                    success=False,
                    previous_level=self.config.current_level,
                    new_level=self.config.current_level,
                    message="Request not found",
                )

            request = self._pending_requests.pop(request_id)
            previous = self.config.current_level

            if not approve:
                return LevelChangeResult(
                    success=False,
                    previous_level=previous,
                    new_level=previous,
                    message="Level change rejected",
                )

            # Apply approved change
            request.approved = True
            request.approved_by = approved_by
            request.approval_timestamp = datetime.utcnow()

            self.config.current_level = request.requested_level
            result = LevelChangeResult(
                success=True,
                previous_level=previous,
                new_level=request.requested_level,
                message=f"Level change approved and applied",
            )

            self._change_history.append(result)

            if self._on_level_change:
                self._on_level_change(result)

            return result

    def opt_in_raw_events(self, confirmation: str) -> bool:
        """
        Opt-in to RAW_ORDER_EVENTS telemetry.

        Requires explicit confirmation and enterprise tier.

        Args:
            confirmation: Must be "I UNDERSTAND RAW ORDER EVENTS WILL BE SENT"

        Returns:
            True if opt-in successful
        """
        if self.config.mode != TelemetryMode.ENTERPRISE:
            return False

        expected = "I UNDERSTAND RAW ORDER EVENTS WILL BE SENT"
        if confirmation != expected:
            return False

        self.config.raw_events_opt_in = True
        return True

    def opt_out_raw_events(self) -> None:
        """Opt-out of RAW_ORDER_EVENTS telemetry."""
        self.config.raw_events_opt_in = False

        # If currently at RAW level, drop down
        if self.config.current_level == TelemetryLevel.RAW_ORDER_EVENTS:
            self.config.current_level = TelemetryLevel.DETAILED_NON_SENSITIVE

    def filter_telemetry(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter telemetry data based on current level.

        Args:
            data: Raw telemetry data

        Returns:
            Filtered data appropriate for current level
        """
        filter_config = self.get_filter()
        return self._apply_filter(data, filter_config)

    def _apply_filter(
        self,
        data: Dict[str, Any],
        filter_config: TelemetryFilter,
    ) -> Dict[str, Any]:
        """Apply filter configuration to data."""
        result = {}

        for key, value in data.items():
            key_lower = key.lower()

            # Check exclusions first
            if key_lower in filter_config.exclude_fields:
                continue

            # Check if wildcard or explicit include
            if "*" in filter_config.include_fields or key_lower in filter_config.include_fields:
                # Handle order events
                if key_lower in ("orders", "order_events", "trades", "fills"):
                    if filter_config.include_order_events:
                        result[key] = value
                    elif filter_config.aggregate_metrics:
                        # Convert to count instead
                        if isinstance(value, list):
                            result[f"{key}_count"] = len(value)
                    continue

                # Handle position details
                if key_lower in ("positions", "position_details"):
                    if filter_config.include_position_details:
                        result[key] = value
                    else:
                        # Include only summary
                        if isinstance(value, list):
                            result[f"{key}_count"] = len(value)
                    continue

                # Include other fields
                result[key] = value

        return result

    def get_change_history(self, limit: int = 100) -> List[LevelChangeResult]:
        """Get level change history."""
        with self._lock:
            return self._change_history[-limit:]

    def get_pending_requests(self) -> List[LevelChangeRequest]:
        """Get pending level change requests."""
        with self._lock:
            return list(self._pending_requests.values())


# ============================================================================
# Convenience Functions
# ============================================================================

def create_retail_manager() -> TelemetryLevelManager:
    """Create manager for retail deployment."""
    return TelemetryLevelManager(TelemetryLevelConfig(
        mode=TelemetryMode.RETAIL,
        current_level=TelemetryLevel.AGGREGATED,
    ))


def create_enterprise_manager(raw_opt_in: bool = False) -> TelemetryLevelManager:
    """Create manager for enterprise deployment."""
    return TelemetryLevelManager(TelemetryLevelConfig(
        mode=TelemetryMode.ENTERPRISE,
        current_level=TelemetryLevel.DETAILED_NON_SENSITIVE,
        raw_events_opt_in=raw_opt_in,
    ))
