# -*- coding: utf-8 -*-
"""
Tests for DataResidencyManager.

CCEA Phase 8 - Data residency and EU compliance tests.
"""

import pytest
from datetime import datetime

from packages.cloud.governance.residency import (
    DataResidencyManager,
    ResidencyConfig,
    ResidencyPolicy,
    DataRegion,
    ResidencyMode,
    DataLocationInfo,
    EU_REGIONS,
    COUNTRY_TO_REGION,
    DEFAULT_REGION,
)


class TestDataResidencyManagerBasic:
    """Basic residency manager tests."""

    def test_create_manager(self):
        """Test creating residency manager."""
        manager = DataResidencyManager()
        assert manager is not None
        assert manager.config is not None

    def test_create_with_config(self):
        """Test creating with custom config."""
        config = ResidencyConfig(
            default_region=DataRegion.EU_WEST,
            enforce_gdpr_for_eu=True,
        )
        manager = DataResidencyManager(config)

        assert manager.config.default_region == DataRegion.EU_WEST


class TestCountryBasedPolicies:
    """Country-based policy tests."""

    @pytest.mark.parametrize("country,expected_region", [
        ("DE", DataRegion.EU_CENTRAL),
        ("FR", DataRegion.EU_WEST),
        ("GB", DataRegion.UK),
        ("US", DataRegion.US_EAST),
        ("JP", DataRegion.AP_NORTHEAST),
    ])
    def test_country_to_region_mapping(self, country, expected_region):
        """Test country to region mapping."""
        manager = DataResidencyManager()
        policy = manager.create_policy_for_country("ws-123", country)

        assert policy.primary_region == expected_region

    def test_eu_country_gdpr_compliant(self):
        """Test EU countries get GDPR-compliant policies."""
        manager = DataResidencyManager()
        policy = manager.create_policy_for_country("ws-123", "DE")

        assert policy.gdpr_compliant is True
        assert policy.requires_eu_only is True

    def test_non_eu_country_policy(self):
        """Test non-EU country policies."""
        manager = DataResidencyManager()
        policy = manager.create_policy_for_country("ws-123", "US")

        assert policy.gdpr_compliant is False
        assert policy.requires_eu_only is False

    def test_unknown_country_uses_default(self):
        """Test unknown country uses default region."""
        manager = DataResidencyManager()
        policy = manager.create_policy_for_country("ws-123", "XX")

        assert policy.primary_region == DEFAULT_REGION


class TestEnterpriseLocalPolicy:
    """Enterprise local-only policy tests."""

    def test_create_local_only_policy(self):
        """Test creating enterprise local-only policy."""
        manager = DataResidencyManager()
        policy = manager.create_enterprise_local_policy("ws-123")

        assert policy.mode == ResidencyMode.LOCAL_ONLY
        assert policy.telemetry_local is True
        assert policy.requires_local_only is True

    def test_local_policy_prevents_cloud_storage(self):
        """Test local-only policy prevents cloud storage."""
        manager = DataResidencyManager()
        manager.create_enterprise_local_policy("ws-123")

        allowed, reason = manager.can_store_in("ws-123", DataRegion.US_EAST)

        assert allowed is False
        assert "local-only" in reason.lower()


class TestSelectiveExportPolicy:
    """Selective export policy tests."""

    def test_create_selective_export_policy(self):
        """Test creating selective export policy."""
        manager = DataResidencyManager()
        policy = manager.create_selective_export_policy(
            workspace_id="ws-123",
            primary_region=DataRegion.EU_WEST,
            export_to={DataRegion.US_EAST},
        )

        assert policy.mode == ResidencyMode.SELECTIVE
        assert DataRegion.US_EAST in policy.export_allowed_to

    def test_selective_export_allowed(self):
        """Test selective export to allowed regions."""
        manager = DataResidencyManager()
        manager.create_selective_export_policy(
            workspace_id="ws-123",
            primary_region=DataRegion.EU_WEST,
            export_to={DataRegion.US_EAST},
        )

        allowed, _ = manager.can_transfer_to("ws-123", DataRegion.US_EAST)
        assert allowed is True

    def test_selective_export_denied_not_in_list(self):
        """Test selective export denied for non-listed regions."""
        manager = DataResidencyManager()
        manager.create_selective_export_policy(
            workspace_id="ws-123",
            primary_region=DataRegion.EU_WEST,
            export_to={DataRegion.US_EAST},
        )

        allowed, reason = manager.can_transfer_to("ws-123", DataRegion.AP_NORTHEAST)

        assert allowed is False
        assert "not in export allowlist" in reason.lower()


class TestStorageRestrictions:
    """Storage restriction tests."""

    def test_can_store_in_allowed_region(self):
        """Test storage allowed in configured region."""
        manager = DataResidencyManager()
        manager.create_policy_for_country("ws-123", "US")

        allowed, _ = manager.can_store_in("ws-123", DataRegion.US_EAST)
        assert allowed is True

    def test_cannot_store_outside_allowed_regions(self):
        """Test storage denied outside allowed regions."""
        config = ResidencyConfig(allow_cross_region_transfer=False)
        manager = DataResidencyManager(config)
        manager.create_policy_for_country("ws-123", "US")

        allowed, reason = manager.can_store_in("ws-123", DataRegion.AP_NORTHEAST)

        assert allowed is False

    def test_eu_only_restriction(self):
        """Test EU-only restriction enforced."""
        manager = DataResidencyManager()
        manager.create_policy_for_country("ws-123", "DE")

        allowed, reason = manager.can_store_in("ws-123", DataRegion.US_EAST)

        assert allowed is False
        # Region restriction is enforced - could be via allowed_regions or EU-only check
        assert "not in allowed" in reason or "EU-only" in reason


class TestTransferRestrictions:
    """Transfer restriction tests."""

    def test_can_transfer_within_eu(self):
        """Test transfer allowed within EU."""
        manager = DataResidencyManager()
        manager.create_policy_for_country("ws-123", "DE")

        allowed, _ = manager.can_transfer_to("ws-123", DataRegion.EU_WEST)
        assert allowed is True

    def test_cannot_transfer_eu_to_non_eu(self):
        """Test transfer denied from EU to non-EU."""
        manager = DataResidencyManager()
        manager.create_policy_for_country("ws-123", "DE")

        allowed, reason = manager.can_transfer_to("ws-123", DataRegion.US_EAST)

        assert allowed is False
        assert "EU data cannot be transferred" in reason or "not allowed" in reason


class TestTelemetryLocalMode:
    """Telemetry local mode tests."""

    def test_telemetry_local_for_enterprise(self):
        """Test telemetry stays local for enterprise local mode."""
        manager = DataResidencyManager()
        manager.create_enterprise_local_policy("ws-123")

        should_keep_local = manager.should_keep_telemetry_local("ws-123")
        assert should_keep_local is True

    def test_telemetry_cloud_for_standard(self):
        """Test telemetry goes to cloud for standard mode."""
        manager = DataResidencyManager()
        manager.create_policy_for_country("ws-123", "US")

        should_keep_local = manager.should_keep_telemetry_local("ws-123")
        assert should_keep_local is False


class TestPolicyManagement:
    """Policy management tests."""

    def test_get_policy(self):
        """Test getting policy."""
        manager = DataResidencyManager()
        created = manager.create_policy_for_country("ws-123", "US")

        retrieved = manager.get_policy("ws-123")

        assert retrieved is not None
        assert retrieved.id == created.id

    def test_get_or_create_policy(self):
        """Test get or create policy."""
        manager = DataResidencyManager()

        # First call creates
        policy1 = manager.get_or_create_policy("ws-123", "US")
        assert policy1 is not None

        # Second call retrieves
        policy2 = manager.get_or_create_policy("ws-123", "DE")
        assert policy2.id == policy1.id  # Same policy

    def test_update_policy(self):
        """Test updating policy."""
        manager = DataResidencyManager()
        manager.create_policy_for_country("ws-123", "US")

        updated = manager.update_policy(
            workspace_id="ws-123",
            updates={"telemetry_local": True},
        )

        assert updated is not None
        assert updated.telemetry_local is True


class TestStorageRegion:
    """Storage region tests."""

    def test_get_storage_region(self):
        """Test getting storage region for workspace."""
        manager = DataResidencyManager()
        manager.create_policy_for_country("ws-123", "DE")

        region = manager.get_storage_region("ws-123")

        assert region == DataRegion.EU_CENTRAL

    def test_storage_region_default_no_policy(self):
        """Test default storage region when no policy."""
        manager = DataResidencyManager()

        region = manager.get_storage_region("unknown-ws")

        assert region == DEFAULT_REGION


class TestComplianceValidation:
    """Compliance validation tests."""

    def test_validate_compliance_pass(self):
        """Test compliance validation passes."""
        manager = DataResidencyManager()
        manager.create_policy_for_country("ws-123", "DE")

        locations = [
            DataLocationInfo(
                data_type="telemetry",
                current_region=DataRegion.EU_CENTRAL,
                is_compliant=False,
            ),
        ]

        compliant, issues = manager.validate_compliance("ws-123", locations)

        assert compliant is True
        assert len(issues) == 0

    def test_validate_compliance_fail(self):
        """Test compliance validation fails."""
        manager = DataResidencyManager()
        manager.create_policy_for_country("ws-123", "DE")

        locations = [
            DataLocationInfo(
                data_type="telemetry",
                current_region=DataRegion.US_EAST,  # Wrong region!
                is_compliant=False,
            ),
        ]

        compliant, issues = manager.validate_compliance("ws-123", locations)

        assert compliant is False
        assert len(issues) > 0


class TestHelperMethods:
    """Helper method tests."""

    def test_get_recommended_region(self):
        """Test getting recommended region."""
        manager = DataResidencyManager()

        assert manager.get_recommended_region("DE") == DataRegion.EU_CENTRAL
        assert manager.get_recommended_region("US") == DataRegion.US_EAST
        assert manager.get_recommended_region("XX") == DEFAULT_REGION

    def test_is_eu_country(self):
        """Test EU country detection."""
        manager = DataResidencyManager()

        assert manager.is_eu_country("DE") is True
        assert manager.is_eu_country("FR") is True
        assert manager.is_eu_country("GB") is True  # UK in EU regions
        assert manager.is_eu_country("US") is False
        assert manager.is_eu_country("JP") is False


class TestAuditLog:
    """Audit log tests."""

    def test_audit_log_on_policy_create(self):
        """Test audit log entry on policy creation."""
        manager = DataResidencyManager()
        manager.create_policy_for_country("ws-123", "DE")

        log = manager.get_audit_log(workspace_id="ws-123")

        assert len(log) > 0
        assert log[0]["action"] == "policy_created"

    def test_audit_log_filtering(self):
        """Test audit log filtering."""
        manager = DataResidencyManager()
        manager.create_policy_for_country("ws-1", "DE")
        manager.create_policy_for_country("ws-2", "US")

        log1 = manager.get_audit_log(workspace_id="ws-1")
        log2 = manager.get_audit_log(workspace_id="ws-2")

        assert all(e["workspace_id"] == "ws-1" for e in log1)
        assert all(e["workspace_id"] == "ws-2" for e in log2)


class TestResidencyPolicySerialization:
    """Policy serialization tests."""

    def test_policy_to_dict(self):
        """Test policy serialization."""
        policy = ResidencyPolicy(
            workspace_id="ws-123",
            primary_region=DataRegion.EU_CENTRAL,
            mode=ResidencyMode.CLOUD,
        )

        data = policy.to_dict()

        assert data["workspace_id"] == "ws-123"
        assert data["primary_region"] == "eu-central-1"
        assert data["mode"] == "CLOUD"


class TestEURegionsConstant:
    """EU regions constant tests."""

    def test_eu_regions_defined(self):
        """Test EU regions are properly defined."""
        assert DataRegion.EU_WEST in EU_REGIONS
        assert DataRegion.EU_CENTRAL in EU_REGIONS
        assert DataRegion.UK in EU_REGIONS
        assert DataRegion.US_EAST not in EU_REGIONS
