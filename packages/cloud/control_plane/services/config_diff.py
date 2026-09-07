# -*- coding: utf-8 -*-
"""
Config Diff Service - Computes diffs between config blobs.

Phase 3 (3.4): Immutable Config Blobs + digest + diff for approve.

Design Doc:
- Cloud stores config as content-addressed blob (digest)
- Changes create new blob (immutable)
- Agent can get blob by digest and show diff for approval

CLOUD ZONE ONLY.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Set, Tuple, Union
from pathlib import Path


class ChangeType(str, Enum):
    """Type of change in config diff."""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


class ChangeImpact(str, Enum):
    """Impact level of config change."""
    TRADING_IMPACTING = "trading_impacting"
    NON_TRADING_IMPACTING = "non_trading_impacting"
    DOCUMENTATION_ONLY = "documentation_only"


# Fields that are trading-impacting when changed
TRADING_IMPACTING_FIELDS: Final[Set[str]] = {
    "max_position_size",
    "max_drawdown",
    "max_daily_loss",
    "risk_limit",
    "position_limit",
    "leverage",
    "universe",
    "instruments",
    "symbols",
    "trading_pairs",
    "allocation",
    "weight",
    "signal_threshold",
    "entry_threshold",
    "exit_threshold",
    "stop_loss",
    "take_profit",
    "rebalance_threshold",
    "execution_mode",
    "live_trading_enabled",
    "paper_trading",
    "broker_account",
    "broker_id",
}

# Fields that are always safe to change
SAFE_FIELDS: Final[Set[str]] = {
    "description",
    "comment",
    "notes",
    "label",
    "display_name",
    "metadata",
    "tags",
    "version",
    "last_updated",
}


@dataclass
class ConfigChange:
    """Single change in config."""
    path: str  # JSON path: "risk.max_drawdown"
    change_type: ChangeType
    old_value: Any = None
    new_value: Any = None
    impact: ChangeImpact = ChangeImpact.NON_TRADING_IMPACTING

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "change_type": self.change_type.value,
            "old_value": self._serialize_value(self.old_value),
            "new_value": self._serialize_value(self.new_value),
            "impact": self.impact.value,
        }

    def _serialize_value(self, value: Any) -> Any:
        """Serialize value for JSON."""
        if isinstance(value, (dict, list, str, int, float, bool, type(None))):
            return value
        return str(value)

    def format_for_display(self) -> str:
        """Format change for human-readable display."""
        if self.change_type == ChangeType.ADDED:
            return f"+ {self.path}: {self.new_value}"
        elif self.change_type == ChangeType.REMOVED:
            return f"- {self.path}: {self.old_value}"
        elif self.change_type == ChangeType.MODIFIED:
            return f"~ {self.path}: {self.old_value} → {self.new_value}"
        else:
            return f"  {self.path}: {self.old_value}"


@dataclass
class ConfigDiffResult:
    """Result of config diff computation."""
    has_changes: bool = False
    old_digest: str = ""
    new_digest: str = ""
    changes: List[ConfigChange] = field(default_factory=list)
    summary: str = ""
    overall_impact: ChangeImpact = ChangeImpact.NON_TRADING_IMPACTING
    trading_impacting_changes: List[str] = field(default_factory=list)
    safe_changes: List[str] = field(default_factory=list)
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def added_count(self) -> int:
        """Count of added fields."""
        return sum(1 for c in self.changes if c.change_type == ChangeType.ADDED)

    @property
    def removed_count(self) -> int:
        """Count of removed fields."""
        return sum(1 for c in self.changes if c.change_type == ChangeType.REMOVED)

    @property
    def modified_count(self) -> int:
        """Count of modified fields."""
        return sum(1 for c in self.changes if c.change_type == ChangeType.MODIFIED)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "has_changes": self.has_changes,
            "old_digest": self.old_digest,
            "new_digest": self.new_digest,
            "changes": [c.to_dict() for c in self.changes],
            "summary": self.summary,
            "overall_impact": self.overall_impact.value,
            "trading_impacting_changes": self.trading_impacting_changes,
            "safe_changes": self.safe_changes,
            "added_count": self.added_count,
            "removed_count": self.removed_count,
            "modified_count": self.modified_count,
            "computed_at": self.computed_at.isoformat(),
        }

    def format_summary(self) -> str:
        """Format summary for human-readable display."""
        lines = []
        lines.append(f"Config Diff: {self.old_digest[:16]}... → {self.new_digest[:16]}...")
        lines.append(f"Impact: {self.overall_impact.value.upper()}")
        lines.append(f"Changes: +{self.added_count} -{self.removed_count} ~{self.modified_count}")
        lines.append("")

        if self.trading_impacting_changes:
            lines.append("⚠️  TRADING IMPACTING CHANGES:")
            for path in self.trading_impacting_changes:
                change = next((c for c in self.changes if c.path == path), None)
                if change:
                    lines.append(f"   {change.format_for_display()}")
            lines.append("")

        if self.safe_changes:
            lines.append("✓ Safe changes:")
            for path in self.safe_changes[:5]:  # Show first 5
                change = next((c for c in self.changes if c.path == path), None)
                if change:
                    lines.append(f"   {change.format_for_display()}")
            if len(self.safe_changes) > 5:
                lines.append(f"   ... and {len(self.safe_changes) - 5} more")

        return "\n".join(lines)


class ConfigDiffService:
    """
    Computes diffs between config blobs.

    Phase 3 (3.4): Enables approval workflow with clear visibility
    into what's changing.

    Usage:
        service = ConfigDiffService()

        # Compute diff
        result = service.compute_diff(old_config, new_config)

        # Show to operator for approval
        print(result.format_summary())

        if result.overall_impact == ChangeImpact.TRADING_IMPACTING:
            # Require explicit approval
            pass
    """

    def __init__(
        self,
        trading_impacting_fields: Optional[Set[str]] = None,
        safe_fields: Optional[Set[str]] = None,
    ):
        """
        Initialize diff service.

        Args:
            trading_impacting_fields: Override trading-impacting field set
            safe_fields: Override safe field set
        """
        self._trading_fields = trading_impacting_fields or TRADING_IMPACTING_FIELDS
        self._safe_fields = safe_fields or SAFE_FIELDS

    def compute_diff(
        self,
        old_config: Optional[Dict[str, Any]],
        new_config: Dict[str, Any],
    ) -> ConfigDiffResult:
        """
        Compute diff between two configs.

        Args:
            old_config: Previous config (None for new config)
            new_config: New config

        Returns:
            ConfigDiffResult with changes
        """
        result = ConfigDiffResult()

        # Compute digests
        result.new_digest = self._compute_digest(new_config)
        if old_config:
            result.old_digest = self._compute_digest(old_config)
        else:
            result.old_digest = ""

        # Handle new config case
        if old_config is None:
            result.has_changes = True
            self._add_all_as_new(new_config, "", result)
        else:
            # Compute actual diff
            self._compute_diff_recursive(old_config, new_config, "", result)

        result.has_changes = len(result.changes) > 0

        # Classify changes
        self._classify_changes(result)

        # Generate summary
        result.summary = self._generate_summary(result)

        return result

    def compute_diff_from_digests(
        self,
        old_digest: str,
        new_digest: str,
        old_config: Dict[str, Any],
        new_config: Dict[str, Any],
    ) -> ConfigDiffResult:
        """
        Compute diff with pre-computed digests.

        Args:
            old_digest: Old config digest
            new_digest: New config digest
            old_config: Old config content
            new_config: New config content

        Returns:
            ConfigDiffResult
        """
        # Quick check: same digest means no changes
        if old_digest == new_digest:
            return ConfigDiffResult(
                has_changes=False,
                old_digest=old_digest,
                new_digest=new_digest,
                summary="No changes (identical digests)",
            )

        result = self.compute_diff(old_config, new_config)
        result.old_digest = old_digest
        result.new_digest = new_digest
        return result

    def _compute_digest(self, config: Dict[str, Any]) -> str:
        """Compute SHA256 digest of config."""
        content = json.dumps(config, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    def _compute_diff_recursive(
        self,
        old: Any,
        new: Any,
        path: str,
        result: ConfigDiffResult,
    ) -> None:
        """Recursively compute diff."""
        if type(old) != type(new):
            # Type changed - treat as modification
            result.changes.append(ConfigChange(
                path=path,
                change_type=ChangeType.MODIFIED,
                old_value=old,
                new_value=new,
            ))
            return

        if isinstance(old, dict) and isinstance(new, dict):
            # Compare dictionaries
            all_keys = set(old.keys()) | set(new.keys())

            for key in sorted(all_keys):
                new_path = f"{path}.{key}" if path else key

                if key not in old:
                    # Added
                    result.changes.append(ConfigChange(
                        path=new_path,
                        change_type=ChangeType.ADDED,
                        new_value=new[key],
                    ))
                elif key not in new:
                    # Removed
                    result.changes.append(ConfigChange(
                        path=new_path,
                        change_type=ChangeType.REMOVED,
                        old_value=old[key],
                    ))
                else:
                    # Recurse
                    self._compute_diff_recursive(
                        old[key], new[key], new_path, result
                    )

        elif isinstance(old, list) and isinstance(new, list):
            # Compare lists element by element
            if old != new:
                result.changes.append(ConfigChange(
                    path=path,
                    change_type=ChangeType.MODIFIED,
                    old_value=old,
                    new_value=new,
                ))

        else:
            # Scalar comparison
            if old != new:
                result.changes.append(ConfigChange(
                    path=path,
                    change_type=ChangeType.MODIFIED,
                    old_value=old,
                    new_value=new,
                ))

    def _add_all_as_new(
        self,
        config: Any,
        path: str,
        result: ConfigDiffResult,
    ) -> None:
        """Add all fields as new (for initial config)."""
        if isinstance(config, dict):
            for key, value in config.items():
                new_path = f"{path}.{key}" if path else key
                result.changes.append(ConfigChange(
                    path=new_path,
                    change_type=ChangeType.ADDED,
                    new_value=value,
                ))
        else:
            result.changes.append(ConfigChange(
                path=path or "root",
                change_type=ChangeType.ADDED,
                new_value=config,
            ))

    def _classify_changes(self, result: ConfigDiffResult) -> None:
        """Classify changes by impact level."""
        has_trading_impact = False

        for change in result.changes:
            # Get field name (last part of path)
            field_name = change.path.split(".")[-1].lower()

            # Check if trading-impacting
            is_trading_impacting = False
            for tf in self._trading_fields:
                if tf.lower() in field_name or field_name in tf.lower():
                    is_trading_impacting = True
                    break

            # Check if safe
            is_safe = field_name in {f.lower() for f in self._safe_fields}

            if is_trading_impacting and not is_safe:
                change.impact = ChangeImpact.TRADING_IMPACTING
                result.trading_impacting_changes.append(change.path)
                has_trading_impact = True
            elif is_safe:
                change.impact = ChangeImpact.DOCUMENTATION_ONLY
                result.safe_changes.append(change.path)
            else:
                # Default to non-trading impacting
                change.impact = ChangeImpact.NON_TRADING_IMPACTING
                result.safe_changes.append(change.path)

        # Set overall impact
        if has_trading_impact:
            result.overall_impact = ChangeImpact.TRADING_IMPACTING
        elif all(c.impact == ChangeImpact.DOCUMENTATION_ONLY for c in result.changes):
            result.overall_impact = ChangeImpact.DOCUMENTATION_ONLY
        else:
            result.overall_impact = ChangeImpact.NON_TRADING_IMPACTING

    def _generate_summary(self, result: ConfigDiffResult) -> str:
        """Generate human-readable summary."""
        if not result.has_changes:
            return "No changes"

        parts = []
        if result.added_count:
            parts.append(f"{result.added_count} added")
        if result.removed_count:
            parts.append(f"{result.removed_count} removed")
        if result.modified_count:
            parts.append(f"{result.modified_count} modified")

        summary = ", ".join(parts)

        if result.trading_impacting_changes:
            summary += f" ({len(result.trading_impacting_changes)} TRADING IMPACTING)"

        return summary


def compute_config_diff(
    old_config: Optional[Dict[str, Any]],
    new_config: Dict[str, Any],
) -> ConfigDiffResult:
    """
    Convenience function to compute config diff.

    Args:
        old_config: Previous config
        new_config: New config

    Returns:
        ConfigDiffResult
    """
    service = ConfigDiffService()
    return service.compute_diff(old_config, new_config)


def is_trading_impacting_change(
    old_config: Optional[Dict[str, Any]],
    new_config: Dict[str, Any],
) -> bool:
    """
    Quick check if config change is trading-impacting.

    Args:
        old_config: Previous config
        new_config: New config

    Returns:
        True if any trading-impacting changes
    """
    result = compute_config_diff(old_config, new_config)
    return result.overall_impact == ChangeImpact.TRADING_IMPACTING
