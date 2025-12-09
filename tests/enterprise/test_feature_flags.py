# -*- coding: utf-8 -*-
"""
Comprehensive tests for Feature Flag Service.

Tests feature flag system for enterprise features per DORA requirements.
"""

from __future__ import annotations

import pytest
from datetime import datetime

from services.enterprise.feature_flags import (
    # Enums
    FeatureTier,
    FeatureStatus,
    RolloutStrategy,
    # Data structures
    FeatureFlag,
    FeatureGate,
    ClientFeatureAccess,
    FeatureFlagConfig,
    # Service
    FeatureFlagService,
    # Factory
    create_feature_flag_service,
)


# =============================================================================
# FeatureFlag Tests
# =============================================================================


class TestFeatureFlag:
    """Tests for FeatureFlag dataclass."""

    def test_create_feature_flag(self) -> None:
        """Test creating a feature flag."""
        flag = FeatureFlag(
            flag_id="flag-1",
            name="Extended Reporting",
            description="PDF/JSON incident reports",
            minimum_tier=FeatureTier.PROFESSIONAL,
            status=FeatureStatus.ENABLED,
            rollout_strategy=RolloutStrategy.ALL_AT_ONCE,
        )
        assert flag.flag_id == "flag-1"
        assert flag.minimum_tier == FeatureTier.PROFESSIONAL
        assert flag.status == FeatureStatus.ENABLED

    def test_is_enabled_for_tier_standard(self) -> None:
        """Test tier check for STANDARD tier."""
        flag = FeatureFlag(
            flag_id="flag-1",
            name="Basic Feature",
            description="Available to all",
            minimum_tier=FeatureTier.STANDARD,
            status=FeatureStatus.ENABLED,
            rollout_strategy=RolloutStrategy.ALL_AT_ONCE,
        )
        assert flag.is_enabled_for_tier(FeatureTier.STANDARD) is True
        assert flag.is_enabled_for_tier(FeatureTier.PROFESSIONAL) is True
        assert flag.is_enabled_for_tier(FeatureTier.ENTERPRISE) is True

    def test_is_enabled_for_tier_professional(self) -> None:
        """Test tier check for PROFESSIONAL tier."""
        flag = FeatureFlag(
            flag_id="flag-1",
            name="Pro Feature",
            description="Professional and above",
            minimum_tier=FeatureTier.PROFESSIONAL,
            status=FeatureStatus.ENABLED,
            rollout_strategy=RolloutStrategy.ALL_AT_ONCE,
        )
        assert flag.is_enabled_for_tier(FeatureTier.STANDARD) is False
        assert flag.is_enabled_for_tier(FeatureTier.PROFESSIONAL) is True
        assert flag.is_enabled_for_tier(FeatureTier.ENTERPRISE) is True

    def test_is_enabled_for_tier_enterprise(self) -> None:
        """Test tier check for ENTERPRISE tier."""
        flag = FeatureFlag(
            flag_id="flag-1",
            name="Enterprise Feature",
            description="Enterprise only",
            minimum_tier=FeatureTier.ENTERPRISE,
            status=FeatureStatus.ENABLED,
            rollout_strategy=RolloutStrategy.ALL_AT_ONCE,
        )
        assert flag.is_enabled_for_tier(FeatureTier.STANDARD) is False
        assert flag.is_enabled_for_tier(FeatureTier.PROFESSIONAL) is False
        assert flag.is_enabled_for_tier(FeatureTier.ENTERPRISE) is True

    def test_is_enabled_for_tier_disabled(self) -> None:
        """Test tier check for disabled flag."""
        flag = FeatureFlag(
            flag_id="flag-1",
            name="Disabled Feature",
            description="This is disabled",
            minimum_tier=FeatureTier.STANDARD,
            status=FeatureStatus.DISABLED,
            rollout_strategy=RolloutStrategy.ALL_AT_ONCE,
        )
        assert flag.is_enabled_for_tier(FeatureTier.STANDARD) is False
        assert flag.is_enabled_for_tier(FeatureTier.ENTERPRISE) is False

    def test_is_enabled_for_tier_deprecated(self) -> None:
        """Test tier check for deprecated flag."""
        flag = FeatureFlag(
            flag_id="flag-1",
            name="Deprecated Feature",
            description="Scheduled for removal",
            minimum_tier=FeatureTier.STANDARD,
            status=FeatureStatus.DEPRECATED,
            rollout_strategy=RolloutStrategy.ALL_AT_ONCE,
        )
        assert flag.is_enabled_for_tier(FeatureTier.STANDARD) is False

    def test_is_enabled_for_tier_internal(self) -> None:
        """Test tier check for internal tier."""
        flag = FeatureFlag(
            flag_id="flag-1",
            name="Internal Feature",
            description="Internal testing only",
            minimum_tier=FeatureTier.INTERNAL,
            status=FeatureStatus.ENABLED,
            rollout_strategy=RolloutStrategy.ALL_AT_ONCE,
        )
        assert flag.is_enabled_for_tier(FeatureTier.ENTERPRISE) is False
        assert flag.is_enabled_for_tier(FeatureTier.INTERNAL) is True

    def test_feature_flag_defaults(self) -> None:
        """Test feature flag default values."""
        flag = FeatureFlag(
            flag_id="flag-1",
            name="Test",
            description="Test",
            minimum_tier=FeatureTier.STANDARD,
            status=FeatureStatus.ENABLED,
            rollout_strategy=RolloutStrategy.ALL_AT_ONCE,
        )
        assert flag.rollout_percentage == 100.0
        assert flag.allowlist == []
        assert flag.blocklist == []
        assert flag.created_by == "system"
        assert flag.dependencies == []


# =============================================================================
# FeatureGate Tests
# =============================================================================


class TestFeatureGate:
    """Tests for FeatureGate dataclass."""

    def test_create_feature_gate(self) -> None:
        """Test creating a feature gate result."""
        gate = FeatureGate(
            flag_id="flag-1",
            client_id="client-1",
            is_enabled=True,
            reason="Feature enabled",
        )
        assert gate.flag_id == "flag-1"
        assert gate.client_id == "client-1"
        assert gate.is_enabled is True
        assert gate.reason == "Feature enabled"

    def test_feature_gate_with_tier(self) -> None:
        """Test feature gate with tier info."""
        gate = FeatureGate(
            flag_id="flag-1",
            client_id="client-1",
            is_enabled=True,
            reason="Feature enabled",
            tier=FeatureTier.ENTERPRISE,
        )
        assert gate.tier == FeatureTier.ENTERPRISE

    def test_feature_gate_with_rollout_bucket(self) -> None:
        """Test feature gate with rollout bucket."""
        gate = FeatureGate(
            flag_id="flag-1",
            client_id="client-1",
            is_enabled=True,
            reason="Percentage rollout",
            rollout_bucket=42.5,
        )
        assert gate.rollout_bucket == 42.5


# =============================================================================
# ClientFeatureAccess Tests
# =============================================================================


class TestClientFeatureAccess:
    """Tests for ClientFeatureAccess dataclass."""

    def test_create_client_access(self) -> None:
        """Test creating client access config."""
        access = ClientFeatureAccess(
            client_id="client-1",
            tier=FeatureTier.PROFESSIONAL,
        )
        assert access.client_id == "client-1"
        assert access.tier == FeatureTier.PROFESSIONAL
        assert access.beta_participant is False

    def test_client_access_with_overrides(self) -> None:
        """Test client access with feature overrides."""
        access = ClientFeatureAccess(
            client_id="client-1",
            tier=FeatureTier.STANDARD,
            custom_features=["enterprise_feature_1"],
            disabled_features=["standard_feature_2"],
            beta_participant=True,
        )
        assert "enterprise_feature_1" in access.custom_features
        assert "standard_feature_2" in access.disabled_features
        assert access.beta_participant is True


# =============================================================================
# FeatureFlagService Tests
# =============================================================================


class TestFeatureFlagService:
    """Tests for FeatureFlagService."""

    def test_create_service_default_config(self) -> None:
        """Test creating service with default config."""
        service = FeatureFlagService()
        assert service.config.default_tier == FeatureTier.STANDARD
        assert service.config.enable_analytics is True

    def test_create_service_custom_config(self) -> None:
        """Test creating service with custom config."""
        config = FeatureFlagConfig(
            cache_ttl_seconds=120,
            enable_analytics=False,
            default_tier=FeatureTier.PROFESSIONAL,
        )
        service = FeatureFlagService(config)
        assert service.config.cache_ttl_seconds == 120
        assert service.config.enable_analytics is False

    def test_default_flags_initialized(self) -> None:
        """Test that default flags are initialized."""
        service = FeatureFlagService()

        # Check some default flags exist
        assert service.get_flag("extended_reporting") is not None
        assert service.get_flag("client_metrics") is not None
        assert service.get_flag("siem_integration") is not None
        assert service.get_flag("tlpt_support") is not None
        assert service.get_flag("multi_region") is not None

    def test_create_flag(self) -> None:
        """Test creating a new flag."""
        service = FeatureFlagService()
        flag = service.create_flag(
            name="Custom Feature",
            description="A custom feature",
            minimum_tier=FeatureTier.PROFESSIONAL,
            status=FeatureStatus.ENABLED,
        )
        assert flag.name == "Custom Feature"
        assert flag.minimum_tier == FeatureTier.PROFESSIONAL

    def test_get_flag(self) -> None:
        """Test getting a flag by ID."""
        service = FeatureFlagService()
        flag = service.get_flag("extended_reporting")
        assert flag is not None
        assert flag.flag_id == "extended_reporting"

    def test_get_flag_not_found(self) -> None:
        """Test getting non-existent flag."""
        service = FeatureFlagService()
        assert service.get_flag("nonexistent") is None

    def test_list_flags(self) -> None:
        """Test listing all flags."""
        service = FeatureFlagService()
        flags = service.list_flags()
        assert len(flags) >= 7  # Default flags

    def test_list_flags_by_status(self) -> None:
        """Test listing flags by status."""
        service = FeatureFlagService()
        enabled_flags = service.list_flags(status=FeatureStatus.ENABLED)
        assert all(f.status == FeatureStatus.ENABLED for f in enabled_flags)

    def test_list_flags_by_tier(self) -> None:
        """Test listing flags by tier."""
        service = FeatureFlagService()
        enterprise_flags = service.list_flags(tier=FeatureTier.ENTERPRISE)
        # All returned flags should be available to enterprise tier
        for flag in enterprise_flags:
            assert flag.is_enabled_for_tier(FeatureTier.ENTERPRISE)

    def test_update_flag_status(self) -> None:
        """Test updating flag status."""
        service = FeatureFlagService()
        flag = service.update_flag(
            "extended_reporting",
            status=FeatureStatus.BETA,
        )
        assert flag is not None
        assert flag.status == FeatureStatus.BETA

    def test_update_flag_rollout_percentage(self) -> None:
        """Test updating flag rollout percentage."""
        service = FeatureFlagService()
        flag = service.update_flag(
            "extended_reporting",
            rollout_percentage=50.0,
        )
        assert flag is not None
        assert flag.rollout_percentage == 50.0

    def test_update_flag_allowlist(self) -> None:
        """Test updating flag allowlist."""
        service = FeatureFlagService()
        flag = service.update_flag(
            "dedicated_region",
            allowlist=["client-1", "client-2"],
        )
        assert flag is not None
        assert "client-1" in flag.allowlist

    def test_update_flag_blocklist(self) -> None:
        """Test updating flag blocklist."""
        service = FeatureFlagService()
        flag = service.update_flag(
            "extended_reporting",
            blocklist=["blocked-client"],
        )
        assert flag is not None
        assert "blocked-client" in flag.blocklist

    def test_update_flag_not_found(self) -> None:
        """Test updating non-existent flag."""
        service = FeatureFlagService()
        assert service.update_flag("nonexistent", status=FeatureStatus.DISABLED) is None

    def test_set_client_access(self) -> None:
        """Test setting client access."""
        service = FeatureFlagService()
        access = service.set_client_access(
            client_id="client-1",
            tier=FeatureTier.ENTERPRISE,
            beta_participant=True,
        )
        assert access.client_id == "client-1"
        assert access.tier == FeatureTier.ENTERPRISE
        assert access.beta_participant is True

    def test_get_client_access(self) -> None:
        """Test getting client access."""
        service = FeatureFlagService()
        service.set_client_access("client-1", FeatureTier.PROFESSIONAL)

        access = service.get_client_access("client-1")
        assert access is not None
        assert access.tier == FeatureTier.PROFESSIONAL

    def test_get_client_access_not_found(self) -> None:
        """Test getting non-existent client access."""
        service = FeatureFlagService()
        assert service.get_client_access("nonexistent") is None

    def test_evaluate_enabled(self) -> None:
        """Test evaluating flag that should be enabled."""
        service = FeatureFlagService()
        service.set_client_access("client-1", FeatureTier.PROFESSIONAL)

        gate = service.evaluate("extended_reporting", "client-1")
        assert gate.is_enabled is True
        assert gate.tier == FeatureTier.PROFESSIONAL

    def test_evaluate_disabled_by_tier(self) -> None:
        """Test evaluating flag disabled by tier."""
        service = FeatureFlagService()
        service.set_client_access("client-1", FeatureTier.STANDARD)

        # siem_integration requires ENTERPRISE tier
        gate = service.evaluate("siem_integration", "client-1")
        assert gate.is_enabled is False
        assert "enterprise" in gate.reason.lower()

    def test_evaluate_flag_not_found(self) -> None:
        """Test evaluating non-existent flag."""
        service = FeatureFlagService()
        gate = service.evaluate("nonexistent", "client-1")
        assert gate.is_enabled is False
        assert "not found" in gate.reason.lower()

    def test_evaluate_client_in_blocklist(self) -> None:
        """Test evaluating for blocked client."""
        service = FeatureFlagService()
        service.set_client_access("blocked-client", FeatureTier.ENTERPRISE)
        service.update_flag("extended_reporting", blocklist=["blocked-client"])

        gate = service.evaluate("extended_reporting", "blocked-client")
        assert gate.is_enabled is False
        assert "blocklist" in gate.reason.lower()

    def test_evaluate_client_disabled_feature(self) -> None:
        """Test evaluating disabled feature for client."""
        service = FeatureFlagService()
        service.set_client_access(
            "client-1",
            FeatureTier.ENTERPRISE,
            disabled_features=["extended_reporting"],
        )

        gate = service.evaluate("extended_reporting", "client-1")
        assert gate.is_enabled is False
        assert "disabled for client" in gate.reason.lower()

    def test_evaluate_flag_disabled(self) -> None:
        """Test evaluating globally disabled flag."""
        service = FeatureFlagService()
        service.set_client_access("client-1", FeatureTier.ENTERPRISE)
        service.update_flag("extended_reporting", status=FeatureStatus.DISABLED)

        gate = service.evaluate("extended_reporting", "client-1")
        assert gate.is_enabled is False
        assert "disabled" in gate.reason.lower()

    def test_evaluate_flag_deprecated(self) -> None:
        """Test evaluating deprecated flag."""
        service = FeatureFlagService()
        service.set_client_access("client-1", FeatureTier.ENTERPRISE)
        service.update_flag("extended_reporting", status=FeatureStatus.DEPRECATED)

        gate = service.evaluate("extended_reporting", "client-1")
        assert gate.is_enabled is False
        assert "deprecated" in gate.reason.lower()

    def test_evaluate_custom_feature_override(self) -> None:
        """Test custom feature override for client."""
        service = FeatureFlagService()
        # Client is STANDARD but has custom access to ENTERPRISE feature
        service.set_client_access(
            "client-1",
            FeatureTier.STANDARD,
            custom_features=["siem_integration"],
        )

        gate = service.evaluate("siem_integration", "client-1")
        assert gate.is_enabled is True
        assert "custom" in gate.reason.lower()

    def test_evaluate_allowlist_strategy_allowed(self) -> None:
        """Test allowlist strategy for allowed client."""
        service = FeatureFlagService()
        service.set_client_access("client-1", FeatureTier.ENTERPRISE)
        service.update_flag("dedicated_region", allowlist=["client-1"])

        gate = service.evaluate("dedicated_region", "client-1")
        assert gate.is_enabled is True
        assert "allowlist" in gate.reason.lower()

    def test_evaluate_allowlist_strategy_not_allowed(self) -> None:
        """Test allowlist strategy for non-allowed client."""
        service = FeatureFlagService()
        service.set_client_access("client-2", FeatureTier.ENTERPRISE)
        service.update_flag("dedicated_region", allowlist=["client-1"])

        gate = service.evaluate("dedicated_region", "client-2")
        assert gate.is_enabled is False
        assert "allowlist" in gate.reason.lower()

    def test_evaluate_beta_participant(self) -> None:
        """Test beta feature for beta participant."""
        service = FeatureFlagService()
        service.set_client_access("client-1", FeatureTier.PROFESSIONAL, beta_participant=True)
        service.update_flag("extended_reporting", status=FeatureStatus.BETA)

        gate = service.evaluate("extended_reporting", "client-1")
        assert gate.is_enabled is True
        assert "beta" in gate.reason.lower()

    def test_evaluate_beta_non_participant(self) -> None:
        """Test beta feature for non-beta participant."""
        service = FeatureFlagService()
        service.set_client_access("client-1", FeatureTier.PROFESSIONAL, beta_participant=False)
        service.update_flag("extended_reporting", status=FeatureStatus.BETA)

        gate = service.evaluate("extended_reporting", "client-1")
        assert gate.is_enabled is False
        assert "beta" in gate.reason.lower()

    def test_evaluate_percentage_rollout(self) -> None:
        """Test percentage rollout strategy."""
        service = FeatureFlagService()
        service.set_client_access("client-1", FeatureTier.PROFESSIONAL)

        flag = service.get_flag("extended_reporting")
        if flag:
            flag.rollout_strategy = RolloutStrategy.PERCENTAGE
            flag.rollout_percentage = 50.0

        gate = service.evaluate("extended_reporting", "client-1")
        # Result depends on hash, but should have rollout bucket
        assert gate.rollout_bucket is not None or gate.is_enabled

    def test_evaluate_default_tier(self) -> None:
        """Test evaluation with default tier (no access set)."""
        service = FeatureFlagService()
        # Don't set client access - should use default STANDARD tier

        gate = service.evaluate("extended_reporting", "unknown-client")
        # extended_reporting requires PROFESSIONAL, so should be disabled
        assert gate.is_enabled is False

    def test_is_enabled_shortcut(self) -> None:
        """Test is_enabled shortcut method."""
        service = FeatureFlagService()
        service.set_client_access("client-1", FeatureTier.ENTERPRISE)

        assert service.is_enabled("extended_reporting", "client-1") is True
        assert service.is_enabled("nonexistent", "client-1") is False

    def test_get_client_features(self) -> None:
        """Test getting all feature states for client."""
        service = FeatureFlagService()
        service.set_client_access("client-1", FeatureTier.ENTERPRISE)

        features = service.get_client_features("client-1")
        assert isinstance(features, dict)
        assert "extended_reporting" in features
        assert "siem_integration" in features

    def test_get_evaluation_stats(self) -> None:
        """Test getting evaluation statistics."""
        service = FeatureFlagService()
        service.set_client_access("client-1", FeatureTier.ENTERPRISE)

        # Perform some evaluations
        service.evaluate("extended_reporting", "client-1")
        service.evaluate("siem_integration", "client-1")
        service.evaluate("nonexistent", "client-1")

        stats = service.get_evaluation_stats()
        assert "total_evaluations" in stats
        assert "enabled_count" in stats
        assert "disabled_count" in stats
        assert "enabled_rate" in stats

    def test_get_evaluation_stats_by_flag(self) -> None:
        """Test getting evaluation stats for specific flag."""
        service = FeatureFlagService()
        service.set_client_access("client-1", FeatureTier.ENTERPRISE)

        # Evaluate same flag multiple times
        for _ in range(5):
            service.evaluate("extended_reporting", "client-1")

        stats = service.get_evaluation_stats(flag_id="extended_reporting")
        assert stats["total_evaluations"] == 5

    def test_get_evaluation_stats_by_client(self) -> None:
        """Test getting evaluation stats for specific client."""
        service = FeatureFlagService()
        service.set_client_access("client-1", FeatureTier.ENTERPRISE)
        service.set_client_access("client-2", FeatureTier.PROFESSIONAL)

        service.evaluate("extended_reporting", "client-1")
        service.evaluate("extended_reporting", "client-2")

        stats = service.get_evaluation_stats(client_id="client-1")
        assert stats["total_evaluations"] == 1


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_feature_flag_service_default(self) -> None:
        """Test creating service with factory function."""
        service = create_feature_flag_service()
        assert isinstance(service, FeatureFlagService)
        assert service.config.default_tier == FeatureTier.STANDARD

    def test_create_feature_flag_service_custom(self) -> None:
        """Test creating service with custom options."""
        service = create_feature_flag_service(
            default_tier=FeatureTier.PROFESSIONAL,
            enable_analytics=False,
        )
        assert service.config.default_tier == FeatureTier.PROFESSIONAL
        assert service.config.enable_analytics is False


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for enum values."""

    def test_feature_tier_values(self) -> None:
        """Test all feature tier values."""
        assert FeatureTier.STANDARD.value == "standard"
        assert FeatureTier.PROFESSIONAL.value == "professional"
        assert FeatureTier.ENTERPRISE.value == "enterprise"
        assert FeatureTier.INTERNAL.value == "internal"

    def test_feature_status_values(self) -> None:
        """Test all feature status values."""
        assert FeatureStatus.DISABLED.value == "disabled"
        assert FeatureStatus.ENABLED.value == "enabled"
        assert FeatureStatus.BETA.value == "beta"
        assert FeatureStatus.DEPRECATED.value == "deprecated"

    def test_rollout_strategy_values(self) -> None:
        """Test all rollout strategy values."""
        assert RolloutStrategy.ALL_AT_ONCE.value == "all_at_once"
        assert RolloutStrategy.PERCENTAGE.value == "percentage"
        assert RolloutStrategy.ALLOWLIST.value == "allowlist"
        assert RolloutStrategy.CANARY.value == "canary"
        assert RolloutStrategy.RING.value == "ring"


# =============================================================================
# Integration Tests
# =============================================================================


class TestFeatureFlagIntegration:
    """Integration tests for feature flag service."""

    def test_full_feature_lifecycle(self) -> None:
        """Test complete feature flag lifecycle."""
        service = FeatureFlagService()

        # Create a new feature
        flag = service.create_flag(
            name="New Feature",
            description="Testing lifecycle",
            minimum_tier=FeatureTier.PROFESSIONAL,
            status=FeatureStatus.DISABLED,
        )

        # Enable for beta
        service.update_flag(flag.flag_id, status=FeatureStatus.BETA)

        # Add beta participant
        service.set_client_access("beta-client", FeatureTier.PROFESSIONAL, beta_participant=True)

        # Check beta access
        assert service.is_enabled(flag.flag_id, "beta-client") is True

        # Regular client shouldn't have access
        service.set_client_access("regular-client", FeatureTier.PROFESSIONAL)
        assert service.is_enabled(flag.flag_id, "regular-client") is False

        # Enable for all
        service.update_flag(flag.flag_id, status=FeatureStatus.ENABLED)
        assert service.is_enabled(flag.flag_id, "regular-client") is True

        # Deprecate
        service.update_flag(flag.flag_id, status=FeatureStatus.DEPRECATED)
        assert service.is_enabled(flag.flag_id, "regular-client") is False

    def test_tiered_access_model(self) -> None:
        """Test tiered access model across all tiers."""
        service = FeatureFlagService()

        # Set up clients for each tier
        service.set_client_access("standard-client", FeatureTier.STANDARD)
        service.set_client_access("pro-client", FeatureTier.PROFESSIONAL)
        service.set_client_access("enterprise-client", FeatureTier.ENTERPRISE)

        # extended_reporting requires PROFESSIONAL
        assert service.is_enabled("extended_reporting", "standard-client") is False
        assert service.is_enabled("extended_reporting", "pro-client") is True
        assert service.is_enabled("extended_reporting", "enterprise-client") is True

        # siem_integration requires ENTERPRISE
        assert service.is_enabled("siem_integration", "standard-client") is False
        assert service.is_enabled("siem_integration", "pro-client") is False
        assert service.is_enabled("siem_integration", "enterprise-client") is True
