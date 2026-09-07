# -*- coding: utf-8 -*-
"""
Layered Policy Engine - Strict policy hierarchy enforcement.

AGENT ZONE ONLY.

Implements the policy hierarchy: LOCAL > CLOUD > MANIFEST
- Local policy always wins (user's machine, user's rules)
- Cloud policy applies where local is not set
- Manifest provides defaults for what's not specified

Design Doc Phase 4.3:
- Policy Firewall + Hard Caps: strict layering LOCAL > CLOUD > MANIFEST
- Cloud cannot override local restrictions
- Agent enforces local rules autonomously
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, Final, List, Optional, Set, Tuple, Union


# Policy source priority (higher = more authority)
class PolicySource(Enum):
    """Sources of policy, in priority order (highest first)."""

    LOCAL = 100  # User's local config - HIGHEST priority
    CLOUD = 50  # Cloud-pushed policy
    MANIFEST = 10  # Strategy manifest defaults
    DEFAULT = 0  # System defaults - LOWEST priority


@dataclass
class PolicyValue:
    """
    A policy value with its source and metadata.
    """

    key: str
    value: Any
    source: PolicySource
    set_at: datetime = field(default_factory=datetime.utcnow)
    reason: str = ""
    locked: bool = False  # If True, cannot be overridden by lower priority

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "source": self.source.name,
            "set_at": self.set_at.isoformat(),
            "reason": self.reason,
            "locked": self.locked,
        }


@dataclass
class PolicyConflict:
    """
    Record of a policy conflict resolution.
    """

    key: str
    winner_source: PolicySource
    winner_value: Any
    loser_source: PolicySource
    loser_value: Any
    resolved_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "winner_source": self.winner_source.name,
            "winner_value": self.winner_value,
            "loser_source": self.loser_source.name,
            "loser_value": self.loser_value,
            "resolved_at": self.resolved_at.isoformat(),
        }


# Policy merge strategies
class MergeStrategy(Enum):
    """How to merge policy values."""

    REPLACE = auto()  # Higher priority replaces lower
    MIN = auto()  # Take minimum (for limits)
    MAX = auto()  # Take maximum (for thresholds)
    UNION = auto()  # Union of lists/sets
    INTERSECT = auto()  # Intersection of lists/sets


# Default merge strategies by key pattern
DEFAULT_MERGE_STRATEGIES: Dict[str, MergeStrategy] = {
    # Risk limits - always take minimum (most restrictive)
    "max_position": MergeStrategy.MIN,
    "max_order_size": MergeStrategy.MIN,
    "max_daily_loss": MergeStrategy.MIN,
    "max_drawdown": MergeStrategy.MIN,
    "max_leverage": MergeStrategy.MIN,
    "max_orders_per_second": MergeStrategy.MIN,
    # Allowlists - intersection (most restrictive)
    "allowed_symbols": MergeStrategy.INTERSECT,
    "allowed_exchanges": MergeStrategy.INTERSECT,
    "egress_allowlist": MergeStrategy.INTERSECT,
    # Blocklists - union (combine all blocks)
    "blocked_symbols": MergeStrategy.UNION,
    "blocked_strategies": MergeStrategy.UNION,
    # Timeouts - take minimum
    "timeout_seconds": MergeStrategy.MIN,
    "max_execution_time": MergeStrategy.MIN,
    # Memory/CPU limits - take minimum
    "max_memory_mb": MergeStrategy.MIN,
    "max_cpu_percent": MergeStrategy.MIN,
}


@dataclass
class LayeredPolicyConfig:
    """
    Configuration for the layered policy engine.
    """

    # Merge strategies by key pattern
    merge_strategies: Dict[str, MergeStrategy] = field(
        default_factory=lambda: DEFAULT_MERGE_STRATEGIES.copy()
    )

    # Keys that local policy can lock (prevent cloud override)
    lockable_keys: Set[str] = field(
        default_factory=lambda: {
            "max_position",
            "max_daily_loss",
            "allowed_symbols",
            "blocked_strategies",
            "kill_switch_triggers",
        }
    )

    # Keys that cloud cannot set at all
    local_only_keys: Set[str] = field(
        default_factory=lambda: {
            "vault_master_key",
            "local_credentials",
            "keychain_config",
            "auto_approve_commands",
        }
    )

    # Audit all policy changes
    audit_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "merge_strategies": {k: v.name for k, v in self.merge_strategies.items()},
            "lockable_keys": list(self.lockable_keys),
            "local_only_keys": list(self.local_only_keys),
            "audit_enabled": self.audit_enabled,
        }


class LayeredPolicyEngine:
    """
    Enforces strict policy hierarchy: LOCAL > CLOUD > MANIFEST.

    SECURITY CRITICAL:
    - Local policy ALWAYS wins over cloud
    - Cloud cannot override user's restrictions
    - Manifest provides safe defaults
    - All conflicts are logged for audit

    Usage:
        engine = LayeredPolicyEngine()

        # Set policies from different sources
        engine.set_local_policy({
            "max_position": 100,
            "allowed_symbols": ["BTC", "ETH"],
        })

        engine.set_cloud_policy({
            "max_position": 500,  # Will be IGNORED (local wins)
            "max_daily_loss": 1000,
        })

        engine.set_manifest_policy({
            "timeout_seconds": 300,
        })

        # Get effective policy
        policy = engine.get_effective_policy()
        # Returns: max_position=100 (local), max_daily_loss=1000 (cloud), timeout=300 (manifest)
    """

    def __init__(
        self,
        config: Optional[LayeredPolicyConfig] = None,
        on_conflict: Optional[Callable[[PolicyConflict], None]] = None,
    ):
        """
        Initialize policy engine.

        Args:
            config: Engine configuration
            on_conflict: Callback when policy conflict occurs
        """
        self.config = config or LayeredPolicyConfig()
        self._on_conflict = on_conflict

        # Policy layers
        self._local: Dict[str, PolicyValue] = {}
        self._cloud: Dict[str, PolicyValue] = {}
        self._manifest: Dict[str, PolicyValue] = {}
        self._defaults: Dict[str, PolicyValue] = {}

        # Conflict log
        self._conflicts: List[PolicyConflict] = []

        # Set system defaults
        self._init_defaults()

    def _init_defaults(self) -> None:
        """Initialize system default policies."""
        defaults = {
            # Risk defaults (conservative)
            "max_position": 1000000,  # $1M max position
            "max_order_size": 100000,  # $100K max order
            "max_daily_loss": 50000,  # $50K max daily loss
            "max_drawdown": 0.10,  # 10% max drawdown
            "max_leverage": 1.0,  # No leverage by default
            # Rate limits
            "max_orders_per_second": 10,
            "max_orders_per_minute": 100,
            # Timeouts
            "timeout_seconds": 30,
            "max_execution_time": 3600,
            # Resource limits
            "max_memory_mb": 512,
            "max_cpu_percent": 100.0,
            # Network
            "network_enabled": False,
            "egress_allowlist": [],
            # Permissions
            "filesystem_readonly": True,
        }

        for key, value in defaults.items():
            self._defaults[key] = PolicyValue(
                key=key,
                value=value,
                source=PolicySource.DEFAULT,
                reason="System default",
            )

    def set_local_policy(
        self,
        policy: Dict[str, Any],
        lock_keys: Optional[Set[str]] = None,
        reason: str = "Local configuration",
    ) -> None:
        """
        Set local policy (highest priority).

        Args:
            policy: Policy key-value pairs
            lock_keys: Keys to lock from cloud override
            reason: Reason for policy change
        """
        for key, value in policy.items():
            self._local[key] = PolicyValue(
                key=key,
                value=value,
                source=PolicySource.LOCAL,
                reason=reason,
                locked=lock_keys is not None and key in lock_keys,
            )

    def set_cloud_policy(
        self,
        policy: Dict[str, Any],
        reason: str = "Cloud policy update",
    ) -> int:
        """
        Set cloud policy (middle priority).

        Args:
            policy: Policy key-value pairs
            reason: Reason for policy change

        Returns:
            Number of policies rejected (due to local-only or locked)
        """
        rejected = 0

        for key, value in policy.items():
            # Check if local-only
            if key in self.config.local_only_keys:
                rejected += 1
                local_pv = self._local.get(key)
                self._log_conflict(
                    key=key,
                    winner_source=PolicySource.LOCAL,
                    winner_value=local_pv.value if local_pv else None,
                    loser_source=PolicySource.CLOUD,
                    loser_value=value,
                )
                continue

            # Check if locked by local
            local_val = self._local.get(key)
            if local_val and local_val.locked:
                rejected += 1
                self._log_conflict(
                    key=key,
                    winner_source=PolicySource.LOCAL,
                    winner_value=local_val.value,
                    loser_source=PolicySource.CLOUD,
                    loser_value=value,
                )
                continue

            self._cloud[key] = PolicyValue(
                key=key,
                value=value,
                source=PolicySource.CLOUD,
                reason=reason,
            )

        return rejected

    def set_manifest_policy(
        self,
        policy: Dict[str, Any],
        reason: str = "Strategy manifest",
    ) -> None:
        """
        Set manifest policy (lowest priority, from artifact).

        Args:
            policy: Policy key-value pairs
            reason: Reason for policy
        """
        for key, value in policy.items():
            self._manifest[key] = PolicyValue(
                key=key,
                value=value,
                source=PolicySource.MANIFEST,
                reason=reason,
            )

    def get_policy(self, key: str) -> Tuple[Any, PolicySource]:
        """
        Get effective policy value for a key.

        Args:
            key: Policy key

        Returns:
            Tuple of (value, source)
        """
        # Check in priority order, applying merge strategies when needed
        local_val = self._local.get(key)
        cloud_val = self._cloud.get(key)

        # If both LOCAL and CLOUD have the key, apply merge strategy
        if local_val and cloud_val:
            merged = self._merge_values(key, local_val, cloud_val)
            return merged, PolicySource.LOCAL

        # Otherwise, return first found in priority order
        if local_val:
            return local_val.value, PolicySource.LOCAL

        if cloud_val:
            return cloud_val.value, PolicySource.CLOUD

        if key in self._manifest:
            return self._manifest[key].value, PolicySource.MANIFEST

        if key in self._defaults:
            return self._defaults[key].value, PolicySource.DEFAULT

        return None, PolicySource.DEFAULT

    def get_effective_policy(self) -> Dict[str, Any]:
        """
        Get the complete effective policy (all keys merged).

        Returns:
            Dictionary of effective policy values
        """
        result: Dict[str, Any] = {}

        # Gather all keys
        all_keys = set()
        all_keys.update(self._defaults.keys())
        all_keys.update(self._manifest.keys())
        all_keys.update(self._cloud.keys())
        all_keys.update(self._local.keys())

        for key in all_keys:
            value, _ = self.get_policy(key)
            result[key] = value

        return result

    def get_policy_audit(self) -> Dict[str, Dict[str, Any]]:
        """
        Get full policy audit showing source of each value.

        Returns:
            Dictionary with policy details and sources
        """
        result: Dict[str, Dict[str, Any]] = {}

        all_keys = set()
        all_keys.update(self._defaults.keys())
        all_keys.update(self._manifest.keys())
        all_keys.update(self._cloud.keys())
        all_keys.update(self._local.keys())

        for key in sorted(all_keys):
            value, source = self.get_policy(key)

            result[key] = {
                "effective_value": value,
                "effective_source": source.name,
                "local": self._local[key].to_dict() if key in self._local else None,
                "cloud": self._cloud[key].to_dict() if key in self._cloud else None,
                "manifest": self._manifest[key].to_dict() if key in self._manifest else None,
                "default": self._defaults[key].to_dict() if key in self._defaults else None,
            }

        return result

    def _merge_values(
        self,
        key: str,
        higher: PolicyValue,
        lower: PolicyValue,
    ) -> Any:
        """Merge two policy values based on strategy."""
        strategy = self._get_merge_strategy(key)

        if strategy == MergeStrategy.REPLACE:
            return higher.value

        elif strategy == MergeStrategy.MIN:
            try:
                return min(higher.value, lower.value)
            except TypeError:
                return higher.value

        elif strategy == MergeStrategy.MAX:
            try:
                return max(higher.value, lower.value)
            except TypeError:
                return higher.value

        elif strategy == MergeStrategy.UNION:
            if isinstance(higher.value, (list, set)) and isinstance(lower.value, (list, set)):
                return list(set(higher.value) | set(lower.value))
            return higher.value

        elif strategy == MergeStrategy.INTERSECT:
            if isinstance(higher.value, (list, set)) and isinstance(lower.value, (list, set)):
                result = list(set(higher.value) & set(lower.value))
                # If intersection is empty but higher has values, use higher (local always wins)
                if not result and higher.value:
                    return higher.value
                return result
            return higher.value

        return higher.value

    def _get_merge_strategy(self, key: str) -> MergeStrategy:
        """Get merge strategy for a key."""
        # Exact match
        if key in self.config.merge_strategies:
            return self.config.merge_strategies[key]

        # Pattern match
        for pattern, strategy in self.config.merge_strategies.items():
            if pattern in key:
                return strategy

        return MergeStrategy.REPLACE

    def _log_conflict(
        self,
        key: str,
        winner_source: PolicySource,
        winner_value: Any,
        loser_source: PolicySource,
        loser_value: Any,
    ) -> None:
        """Log a policy conflict."""
        conflict = PolicyConflict(
            key=key,
            winner_source=winner_source,
            winner_value=winner_value,
            loser_source=loser_source,
            loser_value=loser_value,
        )

        self._conflicts.append(conflict)

        # Trim history
        if len(self._conflicts) > 1000:
            self._conflicts = self._conflicts[-1000:]

        # Callback
        if self._on_conflict:
            self._on_conflict(conflict)

    def get_conflicts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent policy conflicts."""
        return [c.to_dict() for c in self._conflicts[-limit:]]

    def lock_local_key(self, key: str) -> bool:
        """
        Lock a local key from cloud override.

        Args:
            key: Key to lock

        Returns:
            True if locked
        """
        if key not in self._local:
            return False

        if key not in self.config.lockable_keys:
            return False

        self._local[key].locked = True
        return True

    def unlock_local_key(self, key: str) -> bool:
        """
        Unlock a local key (allow cloud override).

        Args:
            key: Key to unlock

        Returns:
            True if unlocked
        """
        if key not in self._local:
            return False

        self._local[key].locked = False
        return True

    def validate_policy(
        self,
        policy: Dict[str, Any],
        source: PolicySource,
    ) -> Tuple[bool, List[str]]:
        """
        Validate a policy before applying.

        Args:
            policy: Policy to validate
            source: Source of the policy

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors: List[str] = []

        for key, value in policy.items():
            # Check local-only restrictions
            if source != PolicySource.LOCAL and key in self.config.local_only_keys:
                errors.append(f"Key '{key}' can only be set locally")

            # Check locked keys
            if source != PolicySource.LOCAL:
                local_val = self._local.get(key)
                if local_val and local_val.locked:
                    errors.append(f"Key '{key}' is locked by local policy")

            # Type validation for numeric limits
            if "max_" in key or "min_" in key:
                if not isinstance(value, (int, float)):
                    errors.append(f"Key '{key}' must be numeric, got {type(value).__name__}")

            # List validation for allowlists
            if "allowlist" in key or "allowed_" in key or "blocked_" in key:
                if not isinstance(value, (list, set)):
                    errors.append(f"Key '{key}' must be a list, got {type(value).__name__}")

        return len(errors) == 0, errors

    def clear_cloud_policy(self) -> None:
        """Clear all cloud policies."""
        self._cloud.clear()

    def clear_manifest_policy(self) -> None:
        """Clear all manifest policies."""
        self._manifest.clear()

    def get_status(self) -> Dict[str, Any]:
        """Get policy engine status."""
        return {
            "local_policies": len(self._local),
            "cloud_policies": len(self._cloud),
            "manifest_policies": len(self._manifest),
            "default_policies": len(self._defaults),
            "locked_keys": [k for k, v in self._local.items() if v.locked],
            "recent_conflicts": len(self._conflicts),
            "audit_enabled": self.config.audit_enabled,
        }


# Convenience function for creating pre-configured engine
def create_policy_engine(
    local_policy: Optional[Dict[str, Any]] = None,
    on_conflict: Optional[Callable[[PolicyConflict], None]] = None,
) -> LayeredPolicyEngine:
    """
    Create a pre-configured policy engine.

    Args:
        local_policy: Initial local policy
        on_conflict: Conflict callback

    Returns:
        Configured LayeredPolicyEngine
    """
    engine = LayeredPolicyEngine(on_conflict=on_conflict)

    if local_policy:
        engine.set_local_policy(local_policy)

    return engine
