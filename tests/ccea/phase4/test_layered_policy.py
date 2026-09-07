# -*- coding: utf-8 -*-
"""
Tests for Layered Policy Engine - Phase 4.3

Tests cover:
- Policy hierarchy (LOCAL > CLOUD > MANIFEST)
- Merge strategies
- Policy locking
- Conflict resolution
"""

import pytest

from packages.agent.policy.layered_policy import (
    LayeredPolicyEngine,
    LayeredPolicyConfig,
    PolicySource,
    PolicyValue,
    PolicyConflict,
    MergeStrategy,
)


class TestPolicySources:
    """Test policy source priorities."""

    def test_source_priority_ordering(self):
        """Test that LOCAL > CLOUD > MANIFEST > DEFAULT."""
        assert PolicySource.LOCAL.value > PolicySource.CLOUD.value
        assert PolicySource.CLOUD.value > PolicySource.MANIFEST.value
        assert PolicySource.MANIFEST.value > PolicySource.DEFAULT.value

    def test_all_sources_exist(self):
        """Test all expected sources exist."""
        sources = [
            PolicySource.LOCAL,
            PolicySource.CLOUD,
            PolicySource.MANIFEST,
            PolicySource.DEFAULT,
        ]
        assert len(sources) == 4


class TestLayeredPolicyEngine:
    """Test LayeredPolicyEngine operations."""

    @pytest.fixture
    def engine(self):
        """Create policy engine."""
        return LayeredPolicyEngine()

    def test_defaults_are_set(self, engine):
        """Test that defaults are initialized."""
        max_position, source = engine.get_policy("max_position")

        assert max_position is not None
        assert source == PolicySource.DEFAULT

    def test_local_overrides_default(self, engine):
        """Test local policy overrides default."""
        engine.set_local_policy({"max_position": 500})

        value, source = engine.get_policy("max_position")

        assert value == 500
        assert source == PolicySource.LOCAL

    def test_local_overrides_cloud(self, engine):
        """Test local policy overrides cloud."""
        engine.set_cloud_policy({"max_position": 1000})
        engine.set_local_policy({"max_position": 500})

        value, source = engine.get_policy("max_position")

        assert value == 500
        assert source == PolicySource.LOCAL

    def test_cloud_overrides_manifest(self, engine):
        """Test cloud policy overrides manifest."""
        engine.set_manifest_policy({"max_position": 1500})
        engine.set_cloud_policy({"max_position": 1000})

        value, source = engine.get_policy("max_position")

        assert value == 1000
        assert source == PolicySource.CLOUD

    def test_local_only_keys_rejected_from_cloud(self, engine):
        """Test local-only keys cannot be set from cloud."""
        rejected = engine.set_cloud_policy({"vault_master_key": "secret"})

        assert rejected == 1
        value, _ = engine.get_policy("vault_master_key")
        assert value != "secret"

    def test_locked_keys_rejected_from_cloud(self, engine):
        """Test locked keys cannot be overridden by cloud."""
        engine.set_local_policy({"max_position": 500}, lock_keys={"max_position"})
        rejected = engine.set_cloud_policy({"max_position": 1000})

        assert rejected == 1
        value, _ = engine.get_policy("max_position")
        assert value == 500

    def test_lock_unlock_key(self, engine):
        """Test locking and unlocking keys."""
        config = LayeredPolicyConfig()
        config.lockable_keys.add("max_position")
        engine = LayeredPolicyEngine(config)

        engine.set_local_policy({"max_position": 500})
        assert engine.lock_local_key("max_position") is True

        # Now cloud cannot override
        rejected = engine.set_cloud_policy({"max_position": 1000})
        assert rejected == 1

        # Unlock and cloud can set
        assert engine.unlock_local_key("max_position") is True
        rejected = engine.set_cloud_policy({"max_position": 1000})
        assert rejected == 0

    def test_effective_policy_combines_all(self, engine):
        """Test effective policy combines all layers."""
        engine.set_local_policy({"local_key": "local_value"})
        engine.set_cloud_policy({"cloud_key": "cloud_value"})
        engine.set_manifest_policy({"manifest_key": "manifest_value"})

        effective = engine.get_effective_policy()

        assert effective["local_key"] == "local_value"
        assert effective["cloud_key"] == "cloud_value"
        assert effective["manifest_key"] == "manifest_value"
        assert "max_position" in effective  # From defaults

    def test_policy_audit_shows_sources(self, engine):
        """Test policy audit shows sources for each key."""
        engine.set_local_policy({"max_position": 500})
        engine.set_cloud_policy({"max_daily_loss": 1000})

        audit = engine.get_policy_audit()

        assert audit["max_position"]["effective_source"] == "LOCAL"
        assert audit["max_daily_loss"]["effective_source"] == "CLOUD"


class TestMergeStrategies:
    """Test policy merge strategies."""

    @pytest.fixture
    def engine(self):
        """Create policy engine."""
        return LayeredPolicyEngine()

    def test_min_merge_for_limits(self, engine):
        """Test MIN merge strategy for limit keys."""
        engine.set_local_policy({"max_position": 500})
        engine.set_cloud_policy({"max_position": 1000})

        value, _ = engine.get_policy("max_position")

        # Should take minimum (most restrictive)
        assert value == 500

    def test_intersect_merge_for_allowlists(self, engine):
        """Test INTERSECT merge strategy for allowlists."""
        engine.set_local_policy({"allowed_symbols": ["BTC", "ETH", "XRP"]})
        engine.set_cloud_policy({"allowed_symbols": ["BTC", "ETH", "SOL"]})

        value, _ = engine.get_policy("allowed_symbols")

        # Should be intersection
        assert set(value) == {"BTC", "ETH"}

    def test_union_merge_for_blocklists(self, engine):
        """Test UNION merge strategy for blocklists."""
        engine.set_local_policy({"blocked_symbols": ["SHIB"]})
        engine.set_cloud_policy({"blocked_symbols": ["DOGE"]})

        value, _ = engine.get_policy("blocked_symbols")

        # Should be union
        assert set(value) == {"SHIB", "DOGE"}


class TestConflictLogging:
    """Test policy conflict logging."""

    def test_conflict_logged_on_rejection(self):
        """Test conflicts are logged when cloud rejected."""
        conflicts_received = []

        def on_conflict(conflict):
            conflicts_received.append(conflict)

        engine = LayeredPolicyEngine(on_conflict=on_conflict)
        engine.set_local_policy({"max_position": 500}, lock_keys={"max_position"})
        engine.set_cloud_policy({"max_position": 1000})

        assert len(conflicts_received) == 1
        assert conflicts_received[0].key == "max_position"
        assert conflicts_received[0].winner_source == PolicySource.LOCAL

    def test_get_conflicts(self):
        """Test retrieving conflict history."""
        engine = LayeredPolicyEngine()
        engine.set_local_policy({"vault_master_key": "local"})
        engine.set_cloud_policy({"vault_master_key": "cloud"})

        conflicts = engine.get_conflicts()

        assert len(conflicts) >= 1


class TestPolicyValidation:
    """Test policy validation."""

    @pytest.fixture
    def engine(self):
        """Create policy engine."""
        return LayeredPolicyEngine()

    def test_validate_local_policy_passes(self, engine):
        """Test valid local policy passes validation."""
        policy = {"max_position": 500, "allowed_symbols": ["BTC"]}

        is_valid, errors = engine.validate_policy(policy, PolicySource.LOCAL)

        assert is_valid is True
        assert errors == []

    def test_validate_cloud_policy_with_local_only_fails(self, engine):
        """Test cloud policy with local-only keys fails."""
        policy = {"vault_master_key": "secret"}

        is_valid, errors = engine.validate_policy(policy, PolicySource.CLOUD)

        assert is_valid is False
        assert any("vault_master_key" in e for e in errors)

    def test_validate_numeric_limits(self, engine):
        """Test numeric limit validation."""
        policy = {"max_position": "not a number"}

        is_valid, errors = engine.validate_policy(policy, PolicySource.LOCAL)

        assert is_valid is False

    def test_validate_list_allowlists(self, engine):
        """Test list validation for allowlists."""
        policy = {"allowed_symbols": "not a list"}

        is_valid, errors = engine.validate_policy(policy, PolicySource.LOCAL)

        assert is_valid is False


class TestPolicyEngineStatus:
    """Test policy engine status reporting."""

    def test_get_status(self):
        """Test status includes expected fields."""
        engine = LayeredPolicyEngine()
        engine.set_local_policy({"max_position": 500})
        engine.set_cloud_policy({"max_daily_loss": 1000})

        status = engine.get_status()

        assert status["local_policies"] >= 1
        assert status["cloud_policies"] >= 1
        assert "locked_keys" in status
        assert "audit_enabled" in status
