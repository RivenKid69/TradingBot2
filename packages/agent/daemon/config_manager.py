# -*- coding: utf-8 -*-
"""
Config Manager - Fetch, diff, and manage configuration blobs.

AGENT ZONE ONLY - handles configuration lifecycle:
- Fetch config blob from Cloud
- Compute digest for verification
- Generate human-readable diff for approval
- Apply config atomically

Design Doc Phase 4:
- Config changes require local approval (TRADING_IMPACTING)
- Cloud cannot push config directly - only via command
- Agent computes and displays diff before approval
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Set, Tuple
from uuid import uuid4

import yaml


# Constants
CONFIG_CACHE_DIR: Final[str] = ".ccea/configs"
MAX_CONFIG_SIZE_KB: Final[int] = 1024  # 1MB max config size


class ConfigChangeType(Enum):
    """Types of config changes."""
    ADDED = auto()
    REMOVED = auto()
    MODIFIED = auto()
    UNCHANGED = auto()


@dataclass
class ConfigChange:
    """
    Single configuration change.
    """
    path: str  # Dot-notation path (e.g., "strategy.risk.max_position")
    change_type: ConfigChangeType
    old_value: Any = None
    new_value: Any = None
    is_sensitive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "change_type": self.change_type.name,
            "old_value": "[REDACTED]" if self.is_sensitive else self.old_value,
            "new_value": "[REDACTED]" if self.is_sensitive else self.new_value,
            "is_sensitive": self.is_sensitive,
        }


@dataclass
class ConfigDiff:
    """
    Diff between two configuration versions.
    """
    old_digest: str
    new_digest: str
    changes: List[ConfigChange] = field(default_factory=list)
    computed_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return len(self.changes) > 0

    @property
    def change_count(self) -> int:
        """Get number of changes."""
        return len(self.changes)

    @property
    def has_sensitive_changes(self) -> bool:
        """Check if any changes involve sensitive values."""
        return any(c.is_sensitive for c in self.changes)

    def get_summary(self) -> str:
        """Get human-readable summary of changes."""
        if not self.has_changes:
            return "No changes"

        added = sum(1 for c in self.changes if c.change_type == ConfigChangeType.ADDED)
        removed = sum(1 for c in self.changes if c.change_type == ConfigChangeType.REMOVED)
        modified = sum(1 for c in self.changes if c.change_type == ConfigChangeType.MODIFIED)

        parts = []
        if added:
            parts.append(f"{added} added")
        if removed:
            parts.append(f"{removed} removed")
        if modified:
            parts.append(f"{modified} modified")

        return ", ".join(parts)

    def get_text_diff(self, context_lines: int = 3) -> str:
        """
        Get unified text diff for display.

        Args:
            context_lines: Number of context lines

        Returns:
            Unified diff string
        """
        lines = []
        lines.append(f"--- old ({self.old_digest[:16]})")
        lines.append(f"+++ new ({self.new_digest[:16]})")
        lines.append("")

        for change in self.changes:
            if change.change_type == ConfigChangeType.ADDED:
                lines.append(f"+ {change.path}: {_format_value(change.new_value, change.is_sensitive)}")
            elif change.change_type == ConfigChangeType.REMOVED:
                lines.append(f"- {change.path}: {_format_value(change.old_value, change.is_sensitive)}")
            elif change.change_type == ConfigChangeType.MODIFIED:
                lines.append(f"  {change.path}:")
                lines.append(f"-   {_format_value(change.old_value, change.is_sensitive)}")
                lines.append(f"+   {_format_value(change.new_value, change.is_sensitive)}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "old_digest": self.old_digest,
            "new_digest": self.new_digest,
            "changes": [c.to_dict() for c in self.changes],
            "summary": self.get_summary(),
            "has_sensitive_changes": self.has_sensitive_changes,
            "computed_at": self.computed_at.isoformat(),
        }


def _format_value(value: Any, is_sensitive: bool) -> str:
    """Format value for display."""
    if is_sensitive:
        return "[REDACTED]"
    if isinstance(value, dict):
        return json.dumps(value, indent=2)
    if isinstance(value, list):
        return json.dumps(value)
    return str(value)


@dataclass
class ConfigVersion:
    """
    A versioned configuration snapshot.
    """
    config_id: str = field(default_factory=lambda: str(uuid4()))
    digest: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    source: str = "local"  # local, cloud, file
    created_at: datetime = field(default_factory=datetime.utcnow)
    applied_at: Optional[datetime] = None
    deployment_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without content for security)."""
        return {
            "config_id": self.config_id,
            "digest": self.digest,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "deployment_id": self.deployment_id,
        }


# Sensitive key patterns for redaction
SENSITIVE_PATTERNS: Final[Set[str]] = {
    "key", "secret", "password", "token", "credential",
    "api_key", "api_secret", "access_token", "private",
    "bearer", "authorization", "auth_token",
}


def _is_sensitive_key(key: str) -> bool:
    """Check if key contains sensitive data."""
    key_lower = key.lower()
    return any(pattern in key_lower for pattern in SENSITIVE_PATTERNS)


def _compute_digest(content: Dict[str, Any]) -> str:
    """Compute SHA-256 digest of config content."""
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dict to dot-notation paths."""
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class ConfigManager:
    """
    Manages configuration lifecycle for the agent.

    SECURITY:
    - All config changes require explicit approval
    - Computes diff before applying
    - Maintains history for audit
    - Cloud cannot directly modify config

    Usage:
        manager = ConfigManager(cache_dir)

        # Set current config
        manager.set_current(config_dict)

        # Receive new config from cloud
        new_version = manager.create_version(new_config, source="cloud")

        # Compute diff
        diff = manager.compute_diff(new_version.config_id)

        # Show diff for approval
        print(diff.get_text_diff())

        # Apply if approved
        if approved:
            manager.apply_version(new_version.config_id)
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_history: int = 50,
    ):
        """
        Initialize config manager.

        Args:
            cache_dir: Directory for config cache
            max_history: Maximum versions to keep in history
        """
        self._cache_dir = cache_dir or Path.home() / CONFIG_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._max_history = max_history

        # Current active config
        self._current: Optional[ConfigVersion] = None

        # Version registry
        self._versions: Dict[str, ConfigVersion] = {}
        self._history: List[str] = []  # config_ids in order

        # Index file
        self._index_path = self._cache_dir / "config_index.json"
        self._load_index()

    def _load_index(self) -> None:
        """Load config index from disk."""
        if not self._index_path.exists():
            return

        try:
            with open(self._index_path) as f:
                data = json.load(f)

            self._history = data.get("history", [])

            # Load current
            if data.get("current_id"):
                current_path = self._cache_dir / f"{data['current_id']}.json"
                if current_path.exists():
                    with open(current_path) as f:
                        content = json.load(f)
                    self._current = ConfigVersion(
                        config_id=data["current_id"],
                        digest=_compute_digest(content),
                        content=content,
                        source="loaded",
                    )
                    self._versions[self._current.config_id] = self._current

        except Exception:
            pass

    def _save_index(self) -> None:
        """Save config index to disk."""
        data = {
            "current_id": self._current.config_id if self._current else None,
            "history": self._history[-self._max_history:],
            "saved_at": datetime.utcnow().isoformat(),
        }

        with open(self._index_path, "w") as f:
            json.dump(data, f, indent=2)

    def _save_version(self, version: ConfigVersion) -> None:
        """Save version content to disk."""
        version_path = self._cache_dir / f"{version.config_id}.json"
        with open(version_path, "w") as f:
            json.dump(version.content, f, indent=2)

    def set_current(self, config: Dict[str, Any], source: str = "local") -> ConfigVersion:
        """
        Set the current active configuration.

        Args:
            config: Configuration dictionary
            source: Source of config

        Returns:
            Created ConfigVersion
        """
        version = self.create_version(config, source=source)
        self._current = version
        version.applied_at = datetime.utcnow()
        self._save_version(version)
        self._save_index()
        return version

    def get_current(self) -> Optional[ConfigVersion]:
        """Get current active configuration."""
        return self._current

    def get_current_config(self) -> Dict[str, Any]:
        """Get current config content."""
        if self._current:
            return copy.deepcopy(self._current.content)
        return {}

    def get_current_digest(self) -> str:
        """Get current config digest."""
        if self._current:
            return self._current.digest
        return ""

    def create_version(
        self,
        config: Dict[str, Any],
        source: str = "cloud",
        deployment_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConfigVersion:
        """
        Create a new config version (not yet applied).

        Args:
            config: Configuration dictionary
            source: Source of config
            deployment_id: Associated deployment
            metadata: Additional metadata

        Returns:
            Created ConfigVersion
        """
        digest = _compute_digest(config)

        version = ConfigVersion(
            digest=digest,
            content=copy.deepcopy(config),
            source=source,
            deployment_id=deployment_id,
            metadata=metadata or {},
        )

        self._versions[version.config_id] = version
        return version

    def get_version(self, config_id: str) -> Optional[ConfigVersion]:
        """Get config version by ID."""
        return self._versions.get(config_id)

    def compute_diff(
        self,
        new_config_id: str,
        old_config_id: Optional[str] = None,
    ) -> ConfigDiff:
        """
        Compute diff between config versions.

        Args:
            new_config_id: ID of new config version
            old_config_id: ID of old version (default: current)

        Returns:
            ConfigDiff with all changes
        """
        new_version = self._versions.get(new_config_id)
        if not new_version:
            raise ValueError(f"Config version not found: {new_config_id}")

        if old_config_id:
            old_version = self._versions.get(old_config_id)
            if not old_version:
                raise ValueError(f"Config version not found: {old_config_id}")
        else:
            old_version = self._current

        old_content = old_version.content if old_version else {}
        new_content = new_version.content

        old_digest = old_version.digest if old_version else ""
        new_digest = new_version.digest

        # Flatten for comparison
        old_flat = _flatten_dict(old_content)
        new_flat = _flatten_dict(new_content)

        all_keys = set(old_flat.keys()) | set(new_flat.keys())
        changes: List[ConfigChange] = []

        for key in sorted(all_keys):
            is_sensitive = _is_sensitive_key(key.split(".")[-1])

            if key not in old_flat:
                changes.append(ConfigChange(
                    path=key,
                    change_type=ConfigChangeType.ADDED,
                    new_value=new_flat[key],
                    is_sensitive=is_sensitive,
                ))
            elif key not in new_flat:
                changes.append(ConfigChange(
                    path=key,
                    change_type=ConfigChangeType.REMOVED,
                    old_value=old_flat[key],
                    is_sensitive=is_sensitive,
                ))
            elif old_flat[key] != new_flat[key]:
                changes.append(ConfigChange(
                    path=key,
                    change_type=ConfigChangeType.MODIFIED,
                    old_value=old_flat[key],
                    new_value=new_flat[key],
                    is_sensitive=is_sensitive,
                ))

        return ConfigDiff(
            old_digest=old_digest,
            new_digest=new_digest,
            changes=changes,
        )

    def apply_version(self, config_id: str) -> bool:
        """
        Apply config version (make it current).

        Args:
            config_id: Config version ID

        Returns:
            True if applied successfully
        """
        version = self._versions.get(config_id)
        if not version:
            return False

        # Archive current
        if self._current and self._current.config_id != config_id:
            self._history.append(self._current.config_id)

        # Apply new
        self._current = version
        version.applied_at = datetime.utcnow()

        self._save_version(version)
        self._save_index()

        # Trim history
        while len(self._history) > self._max_history:
            old_id = self._history.pop(0)
            self._cleanup_version(old_id)

        return True

    def _cleanup_version(self, config_id: str) -> None:
        """Remove old version files."""
        version_path = self._cache_dir / f"{config_id}.json"
        if version_path.exists():
            version_path.unlink(missing_ok=True)
        self._versions.pop(config_id, None)

    def rollback(self) -> Optional[ConfigVersion]:
        """
        Rollback to previous config.

        Returns:
            Previous ConfigVersion if available
        """
        if not self._history:
            return None

        previous_id = self._history.pop()
        previous_path = self._cache_dir / f"{previous_id}.json"

        if not previous_path.exists():
            return self.rollback()  # Try next

        with open(previous_path) as f:
            content = json.load(f)

        version = ConfigVersion(
            config_id=previous_id,
            digest=_compute_digest(content),
            content=content,
            source="rollback",
            applied_at=datetime.utcnow(),
        )

        self._versions[previous_id] = version
        self._current = version
        self._save_index()

        return version

    def validate_config(
        self,
        config: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Validate configuration.

        Args:
            config: Config to validate
            schema: Optional JSON schema

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors: List[str] = []

        # Basic validation
        if not isinstance(config, dict):
            errors.append("Config must be a dictionary")
            return False, errors

        # Size check
        config_str = json.dumps(config)
        if len(config_str) > MAX_CONFIG_SIZE_KB * 1024:
            errors.append(f"Config too large: {len(config_str)} bytes > {MAX_CONFIG_SIZE_KB}KB")

        # Schema validation if provided
        if schema:
            try:
                import jsonschema
                jsonschema.validate(config, schema)
            except ImportError:
                pass  # jsonschema not available
            except Exception as e:
                errors.append(f"Schema validation failed: {e}")

        return len(errors) == 0, errors

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get config history.

        Args:
            limit: Max versions to return

        Returns:
            List of version summaries
        """
        result = []
        for config_id in reversed(self._history[-limit:]):
            version = self._versions.get(config_id)
            if version:
                result.append(version.to_dict())
        return result

    def export_config(
        self,
        config_id: Optional[str] = None,
        format: str = "yaml",
    ) -> str:
        """
        Export config in specified format.

        Args:
            config_id: Config ID (default: current)
            format: Export format (yaml, json)

        Returns:
            Exported config string
        """
        if config_id:
            version = self._versions.get(config_id)
        else:
            version = self._current

        if not version:
            return ""

        if format == "yaml":
            return yaml.dump(version.content, default_flow_style=False)
        else:
            return json.dumps(version.content, indent=2)

    def import_config(
        self,
        content: str,
        format: str = "auto",
        source: str = "import",
    ) -> ConfigVersion:
        """
        Import config from string.

        Args:
            content: Config string
            format: Format (auto, yaml, json)
            source: Source identifier

        Returns:
            Created ConfigVersion
        """
        if format == "auto":
            # Try YAML first (superset of JSON)
            try:
                config = yaml.safe_load(content)
            except Exception:
                config = json.loads(content)
        elif format == "yaml":
            config = yaml.safe_load(content)
        else:
            config = json.loads(content)

        return self.create_version(config, source=source)

    def get_status(self) -> Dict[str, Any]:
        """Get config manager status."""
        return {
            "cache_dir": str(self._cache_dir),
            "current_digest": self._current.digest if self._current else None,
            "current_source": self._current.source if self._current else None,
            "pending_versions": len(self._versions) - (1 if self._current else 0),
            "history_length": len(self._history),
            "max_history": self._max_history,
        }
