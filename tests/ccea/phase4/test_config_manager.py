# -*- coding: utf-8 -*-
"""
Tests for Config Manager - Phase 4.1

Tests cover:
- Config versioning
- Diff computation
- Sensitive value redaction
- Config application and rollback
"""

import tempfile
from pathlib import Path

import pytest

from packages.agent.daemon.config_manager import (
    ConfigManager,
    ConfigVersion,
    ConfigDiff,
    ConfigChange,
    ConfigChangeType,
    _compute_digest,
    _flatten_dict,
    _is_sensitive_key,
)


class TestConfigHelpers:
    """Test config helper functions."""

    def test_compute_digest_deterministic(self):
        """Test digest computation is deterministic."""
        config = {"a": 1, "b": 2, "c": {"nested": "value"}}

        digest1 = _compute_digest(config)
        digest2 = _compute_digest(config)

        assert digest1 == digest2
        assert len(digest1) == 64  # SHA-256 hex

    def test_compute_digest_order_independent(self):
        """Test digest is independent of key order."""
        config1 = {"a": 1, "b": 2}
        config2 = {"b": 2, "a": 1}

        assert _compute_digest(config1) == _compute_digest(config2)

    def test_flatten_dict_simple(self):
        """Test flattening simple dict."""
        d = {"a": 1, "b": 2}
        flat = _flatten_dict(d)

        assert flat == {"a": 1, "b": 2}

    def test_flatten_dict_nested(self):
        """Test flattening nested dict."""
        d = {"level1": {"level2": {"value": 123}}}
        flat = _flatten_dict(d)

        assert flat == {"level1.level2.value": 123}

    def test_is_sensitive_key_matches(self):
        """Test sensitive key detection."""
        sensitive_keys = [
            "api_key",
            "api_secret",
            "password",
            "access_token",
            "auth_token",
            "bearer_token",
        ]

        for key in sensitive_keys:
            assert _is_sensitive_key(key) is True

    def test_is_sensitive_key_non_sensitive(self):
        """Test non-sensitive keys."""
        non_sensitive = [
            "max_position",
            "symbols",
            "exchange",
            "strategy_name",
        ]

        for key in non_sensitive:
            assert _is_sensitive_key(key) is False


class TestConfigVersion:
    """Test ConfigVersion dataclass."""

    def test_creation(self):
        """Test version creation."""
        version = ConfigVersion(
            digest="abc123",
            content={"key": "value"},
            source="local",
        )

        assert version.digest == "abc123"
        assert version.content == {"key": "value"}
        assert version.source == "local"
        assert version.config_id  # Should have UUID

    def test_to_dict(self):
        """Test version serialization."""
        version = ConfigVersion(
            digest="abc123",
            content={"key": "value"},
            source="cloud",
        )

        data = version.to_dict()

        assert "config_id" in data
        assert data["digest"] == "abc123"
        assert data["source"] == "cloud"
        # content is NOT in to_dict for security
        assert "content" not in data


class TestConfigDiff:
    """Test config diff computation."""

    def test_diff_no_changes(self):
        """Test diff with no changes."""
        diff = ConfigDiff(
            old_digest="abc",
            new_digest="abc",
            changes=[],
        )

        assert diff.has_changes is False
        assert diff.change_count == 0
        assert diff.get_summary() == "No changes"

    def test_diff_with_changes(self):
        """Test diff with changes."""
        diff = ConfigDiff(
            old_digest="old",
            new_digest="new",
            changes=[
                ConfigChange(
                    path="added.key",
                    change_type=ConfigChangeType.ADDED,
                    new_value=123,
                ),
                ConfigChange(
                    path="removed.key",
                    change_type=ConfigChangeType.REMOVED,
                    old_value=456,
                ),
                ConfigChange(
                    path="modified.key",
                    change_type=ConfigChangeType.MODIFIED,
                    old_value=1,
                    new_value=2,
                ),
            ],
        )

        assert diff.has_changes is True
        assert diff.change_count == 3
        assert "1 added" in diff.get_summary()
        assert "1 removed" in diff.get_summary()
        assert "1 modified" in diff.get_summary()

    def test_diff_sensitive_detection(self):
        """Test sensitive change detection."""
        diff = ConfigDiff(
            old_digest="old",
            new_digest="new",
            changes=[
                ConfigChange(
                    path="api_key",
                    change_type=ConfigChangeType.MODIFIED,
                    old_value="old_secret",
                    new_value="new_secret",
                    is_sensitive=True,
                ),
            ],
        )

        assert diff.has_sensitive_changes is True

    def test_diff_text_output(self):
        """Test text diff output."""
        diff = ConfigDiff(
            old_digest="old123",
            new_digest="new456",
            changes=[
                ConfigChange(
                    path="value",
                    change_type=ConfigChangeType.ADDED,
                    new_value=100,
                ),
            ],
        )

        text = diff.get_text_diff()

        assert "old123" in text
        assert "new456" in text
        assert "+" in text  # Added indicator


class TestConfigManager:
    """Test ConfigManager operations."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_cache_dir):
        """Create ConfigManager with temp directory."""
        return ConfigManager(cache_dir=temp_cache_dir)

    def test_init(self, manager):
        """Test manager initialization."""
        assert manager.get_current() is None
        assert manager.get_current_config() == {}
        assert manager.get_current_digest() == ""

    def test_set_current(self, manager):
        """Test setting current config."""
        config = {"max_position": 100, "symbols": ["BTC", "ETH"]}

        version = manager.set_current(config, source="local")

        assert version.content == config
        assert version.source == "local"
        assert manager.get_current() == version
        assert manager.get_current_config() == config

    def test_create_version(self, manager):
        """Test creating a new version."""
        config = {"key": "value"}

        version = manager.create_version(config, source="cloud")

        assert version.source == "cloud"
        assert version.content == config
        assert version.applied_at is None  # Not applied yet

    def test_compute_diff_added(self, manager):
        """Test computing diff with additions."""
        manager.set_current({"a": 1}, source="local")

        new_version = manager.create_version({"a": 1, "b": 2}, source="cloud")
        diff = manager.compute_diff(new_version.config_id)

        added_keys = [c.path for c in diff.changes if c.change_type == ConfigChangeType.ADDED]
        assert "b" in added_keys

    def test_compute_diff_removed(self, manager):
        """Test computing diff with removals."""
        manager.set_current({"a": 1, "b": 2}, source="local")

        new_version = manager.create_version({"a": 1}, source="cloud")
        diff = manager.compute_diff(new_version.config_id)

        removed_keys = [c.path for c in diff.changes if c.change_type == ConfigChangeType.REMOVED]
        assert "b" in removed_keys

    def test_compute_diff_modified(self, manager):
        """Test computing diff with modifications."""
        manager.set_current({"a": 1}, source="local")

        new_version = manager.create_version({"a": 2}, source="cloud")
        diff = manager.compute_diff(new_version.config_id)

        modified_keys = [c.path for c in diff.changes if c.change_type == ConfigChangeType.MODIFIED]
        assert "a" in modified_keys

    def test_apply_version(self, manager):
        """Test applying a version."""
        manager.set_current({"old": "config"}, source="local")

        new_version = manager.create_version({"new": "config"}, source="cloud")
        success = manager.apply_version(new_version.config_id)

        assert success is True
        assert manager.get_current_config() == {"new": "config"}

    def test_rollback(self, manager):
        """Test config rollback."""
        manager.set_current({"v1": "config"}, source="local")

        new_version = manager.create_version({"v2": "config"}, source="cloud")
        manager.apply_version(new_version.config_id)

        rolled_back = manager.rollback()

        assert rolled_back is not None
        assert manager.get_current_config() == {"v1": "config"}

    def test_validate_config_valid(self, manager):
        """Test validating valid config."""
        config = {"key": "value", "number": 123}

        is_valid, errors = manager.validate_config(config)

        assert is_valid is True
        assert errors == []

    def test_validate_config_invalid_type(self, manager):
        """Test validating invalid config type."""
        config = "not a dict"

        is_valid, errors = manager.validate_config(config)

        assert is_valid is False
        assert len(errors) > 0

    def test_export_yaml(self, manager):
        """Test exporting config as YAML."""
        manager.set_current({"key": "value"}, source="local")

        yaml_str = manager.export_config(format="yaml")

        assert "key: value" in yaml_str

    def test_export_json(self, manager):
        """Test exporting config as JSON."""
        manager.set_current({"key": "value"}, source="local")

        json_str = manager.export_config(format="json")

        assert '"key": "value"' in json_str

    def test_import_yaml(self, manager):
        """Test importing config from YAML."""
        yaml_str = "key: value\nnumber: 123"

        version = manager.import_config(yaml_str, format="yaml")

        assert version.content == {"key": "value", "number": 123}

    def test_get_status(self, manager):
        """Test getting manager status."""
        status = manager.get_status()

        assert "cache_dir" in status
        assert "current_digest" in status
        assert "pending_versions" in status
