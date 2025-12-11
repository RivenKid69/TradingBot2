# -*- coding: utf-8 -*-
"""
Tests for MiFID II Financial Entity Configuration Module (ARCHIVED).

Comprehensive tests for Pydantic configuration models in the ARCHIVE package.
100% coverage for all configuration classes and validation logic.

These configurations are for Investment Firms only, NOT for ICT Providers.
"""

import pytest
import tempfile
import os
import yaml
import warnings

# Suppress deprecation warning for test imports
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from services.archive.mifid_financial_entity.config import (
        ComplianceMode,
        LEIStatus,
        LEIConfig,
        TransactionReportingConfig,
        NCANotificationConfig,
        GovernanceConfig,
        MiFIDIIComplianceConfig,
        load_mifid_compliance_config,
    )


class TestComplianceMode:
    """Tests for ComplianceMode enum."""

    def test_production_value(self):
        """Test PRODUCTION enum value."""
        assert ComplianceMode.PRODUCTION.value == "production"

    def test_testing_value(self):
        """Test TESTING enum value."""
        assert ComplianceMode.TESTING.value == "testing"

    def test_disabled_value(self):
        """Test DISABLED enum value."""
        assert ComplianceMode.DISABLED.value == "disabled"

    def test_enum_from_string(self):
        """Test creating enum from string."""
        assert ComplianceMode("production") == ComplianceMode.PRODUCTION
        assert ComplianceMode("testing") == ComplianceMode.TESTING
        assert ComplianceMode("disabled") == ComplianceMode.DISABLED


class TestLEIStatus:
    """Tests for LEIStatus enum."""

    def test_issued_value(self):
        """Test ISSUED status."""
        assert LEIStatus.ISSUED.value == "ISSUED"

    def test_lapsed_value(self):
        """Test LAPSED status."""
        assert LEIStatus.LAPSED.value == "LAPSED"

    def test_merged_value(self):
        """Test MERGED status."""
        assert LEIStatus.MERGED.value == "MERGED"

    def test_retired_value(self):
        """Test RETIRED status."""
        assert LEIStatus.RETIRED.value == "RETIRED"

    def test_pending_statuses(self):
        """Test pending statuses."""
        assert LEIStatus.PENDING_TRANSFER.value == "PENDING_TRANSFER"
        assert LEIStatus.PENDING_ARCHIVAL.value == "PENDING_ARCHIVAL"

    def test_all_statuses(self):
        """Test all LEI statuses are accessible."""
        all_statuses = [s.value for s in LEIStatus]
        assert len(all_statuses) == 8
        assert "ISSUED" in all_statuses
        assert "ANNULLED" in all_statuses
        assert "DUPLICATE" in all_statuses


class TestLEIConfig:
    """Tests for LEIConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LEIConfig()
        assert config.own_lei == ""
        assert config.gleif_api_url == "https://api.gleif.org/api/v1"
        assert config.cache_ttl_hours == 24
        assert config.verify_before_trade is True
        assert config.renewal_warning_days == 30
        assert config.allow_pending_status is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LEIConfig(
            own_lei="5493001KJTIIGC8Y1R12",
            cache_ttl_hours=12,
            verify_before_trade=False,
            renewal_warning_days=60,
        )
        assert config.own_lei == "5493001KJTIIGC8Y1R12"
        assert config.cache_ttl_hours == 12
        assert config.verify_before_trade is False
        assert config.renewal_warning_days == 60

    def test_cache_ttl_min(self):
        """Test minimum cache TTL."""
        config = LEIConfig(cache_ttl_hours=1)
        assert config.cache_ttl_hours == 1

    def test_cache_ttl_max(self):
        """Test maximum cache TTL."""
        config = LEIConfig(cache_ttl_hours=168)
        assert config.cache_ttl_hours == 168

    def test_cache_ttl_invalid_min(self):
        """Test cache TTL below minimum."""
        with pytest.raises(ValueError):
            LEIConfig(cache_ttl_hours=0)

    def test_cache_ttl_invalid_max(self):
        """Test cache TTL above maximum."""
        with pytest.raises(ValueError):
            LEIConfig(cache_ttl_hours=200)

    def test_renewal_warning_days_min(self):
        """Test minimum renewal warning days."""
        config = LEIConfig(renewal_warning_days=7)
        assert config.renewal_warning_days == 7

    def test_renewal_warning_days_max(self):
        """Test maximum renewal warning days."""
        config = LEIConfig(renewal_warning_days=90)
        assert config.renewal_warning_days == 90

    def test_renewal_warning_days_invalid(self):
        """Test invalid renewal warning days."""
        with pytest.raises(ValueError):
            LEIConfig(renewal_warning_days=5)

        with pytest.raises(ValueError):
            LEIConfig(renewal_warning_days=100)

    def test_lei_format_valid(self):
        """Test valid LEI format."""
        # Valid LEI: 18 alphanumeric + 2 check digits
        config = LEIConfig(own_lei="529900W18LQJJN6SJ336")
        assert config.own_lei == "529900W18LQJJN6SJ336"

    def test_lei_empty_allowed(self):
        """Test empty LEI is allowed."""
        config = LEIConfig(own_lei="")
        assert config.own_lei == ""

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValueError):
            LEIConfig(unknown_field="value")


class TestTransactionReportingConfig:
    """Tests for TransactionReportingConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TransactionReportingConfig()
        assert config.enabled is True
        assert config.arm_provider == ""
        assert config.arm_environment == "test"
        assert config.report_deadline_hours == 24
        assert config.auto_submit is False
        assert config.batch_size == 100

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TransactionReportingConfig(
            enabled=False,
            arm_provider="UNAVISTA",
            arm_environment="production",
            report_deadline_hours=12,
            auto_submit=True,
            batch_size=500,
        )
        assert config.enabled is False
        assert config.arm_provider == "UNAVISTA"
        assert config.arm_environment == "production"
        assert config.report_deadline_hours == 12
        assert config.auto_submit is True
        assert config.batch_size == 500

    def test_report_deadline_hours_range(self):
        """Test report deadline hours range."""
        config = TransactionReportingConfig(report_deadline_hours=1)
        assert config.report_deadline_hours == 1

        config = TransactionReportingConfig(report_deadline_hours=48)
        assert config.report_deadline_hours == 48

    def test_report_deadline_hours_invalid(self):
        """Test invalid report deadline hours."""
        with pytest.raises(ValueError):
            TransactionReportingConfig(report_deadline_hours=0)

        with pytest.raises(ValueError):
            TransactionReportingConfig(report_deadline_hours=49)

    def test_batch_size_range(self):
        """Test batch size range."""
        config = TransactionReportingConfig(batch_size=1)
        assert config.batch_size == 1

        config = TransactionReportingConfig(batch_size=10000)
        assert config.batch_size == 10000

    def test_batch_size_invalid(self):
        """Test invalid batch size."""
        with pytest.raises(ValueError):
            TransactionReportingConfig(batch_size=0)


class TestNCANotificationConfig:
    """Tests for NCANotificationConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = NCANotificationConfig()
        assert config.jurisdiction == ""
        assert config.notification_email == ""
        assert config.auto_notify is False
        assert "algorithm_deployment" in config.notification_types
        assert "significant_change" in config.notification_types
        assert "incident" in config.notification_types

    def test_custom_values(self):
        """Test custom configuration values."""
        config = NCANotificationConfig(
            jurisdiction="FCA",
            notification_email="compliance@firm.com",
            auto_notify=True,
            notification_types=["algorithm_deployment"],
        )
        assert config.jurisdiction == "FCA"
        assert config.notification_email == "compliance@firm.com"
        assert config.auto_notify is True
        assert len(config.notification_types) == 1

    def test_empty_notification_types(self):
        """Test empty notification types."""
        config = NCANotificationConfig(notification_types=[])
        assert config.notification_types == []


class TestGovernanceConfig:
    """Tests for GovernanceConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GovernanceConfig()
        assert config.policies_path == "docs/compliance/policies"
        assert config.review_frequency_months == 12
        assert config.require_sign_off is True
        assert config.version_control is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = GovernanceConfig(
            policies_path="/custom/policies",
            review_frequency_months=6,
            require_sign_off=False,
            version_control=False,
        )
        assert config.policies_path == "/custom/policies"
        assert config.review_frequency_months == 6
        assert config.require_sign_off is False
        assert config.version_control is False

    def test_review_frequency_range(self):
        """Test review frequency range."""
        config = GovernanceConfig(review_frequency_months=1)
        assert config.review_frequency_months == 1

        config = GovernanceConfig(review_frequency_months=24)
        assert config.review_frequency_months == 24

    def test_review_frequency_invalid(self):
        """Test invalid review frequency."""
        with pytest.raises(ValueError):
            GovernanceConfig(review_frequency_months=0)

        with pytest.raises(ValueError):
            GovernanceConfig(review_frequency_months=25)


class TestMiFIDIIComplianceConfig:
    """Tests for MiFIDIIComplianceConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MiFIDIIComplianceConfig()
        assert config.enabled is True
        assert config.mode == ComplianceMode.TESTING
        assert isinstance(config.lei, LEIConfig)
        assert isinstance(config.transaction_reporting, TransactionReportingConfig)
        assert isinstance(config.nca, NCANotificationConfig)
        assert isinstance(config.governance, GovernanceConfig)

    def test_production_mode_requires_lei(self):
        """Test that production mode requires LEI."""
        with pytest.raises(ValueError, match="LEI is required for production"):
            MiFIDIIComplianceConfig(
                mode=ComplianceMode.PRODUCTION,
                lei=LEIConfig(own_lei=""),
                nca=NCANotificationConfig(jurisdiction="FCA"),
                transaction_reporting=TransactionReportingConfig(enabled=False),
            )

    def test_production_mode_requires_nca(self):
        """Test that production mode requires NCA jurisdiction."""
        with pytest.raises(ValueError, match="NCA jurisdiction is required"):
            MiFIDIIComplianceConfig(
                mode=ComplianceMode.PRODUCTION,
                lei=LEIConfig(own_lei="5493001KJTIIGC8Y1R12"),
                nca=NCANotificationConfig(jurisdiction=""),
                transaction_reporting=TransactionReportingConfig(enabled=False),
            )

    def test_production_mode_requires_arm(self):
        """Test that production mode requires ARM provider when reporting enabled."""
        with pytest.raises(ValueError, match="ARM provider is required"):
            MiFIDIIComplianceConfig(
                mode=ComplianceMode.PRODUCTION,
                lei=LEIConfig(own_lei="5493001KJTIIGC8Y1R12"),
                nca=NCANotificationConfig(jurisdiction="FCA"),
                transaction_reporting=TransactionReportingConfig(
                    enabled=True,
                    arm_provider="",
                ),
            )

    def test_production_mode_valid(self):
        """Test valid production mode configuration."""
        config = MiFIDIIComplianceConfig(
            mode=ComplianceMode.PRODUCTION,
            lei=LEIConfig(own_lei="5493001KJTIIGC8Y1R12"),
            nca=NCANotificationConfig(jurisdiction="FCA"),
            transaction_reporting=TransactionReportingConfig(
                enabled=True,
                arm_provider="UNAVISTA",
            ),
        )
        assert config.mode == ComplianceMode.PRODUCTION

    def test_testing_mode_no_requirements(self):
        """Test that testing mode has no special requirements."""
        config = MiFIDIIComplianceConfig(
            mode=ComplianceMode.TESTING,
            lei=LEIConfig(own_lei=""),
        )
        assert config.mode == ComplianceMode.TESTING

    def test_disabled_mode(self):
        """Test disabled mode configuration."""
        config = MiFIDIIComplianceConfig(
            enabled=False,
            mode=ComplianceMode.DISABLED,
        )
        assert config.enabled is False
        assert config.mode == ComplianceMode.DISABLED

    def test_nested_config(self):
        """Test nested configuration objects."""
        config = MiFIDIIComplianceConfig(
            lei=LEIConfig(
                own_lei="5493001KJTIIGC8Y1R12",
                cache_ttl_hours=12,
            ),
            transaction_reporting=TransactionReportingConfig(
                arm_provider="TRAX",
                batch_size=200,
            ),
            nca=NCANotificationConfig(
                jurisdiction="BaFin",
            ),
        )
        assert config.lei.own_lei == "5493001KJTIIGC8Y1R12"
        assert config.transaction_reporting.arm_provider == "TRAX"
        assert config.nca.jurisdiction == "BaFin"

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed in top-level config."""
        config = MiFIDIIComplianceConfig(custom_field="custom_value")
        # Should not raise - extra fields allowed


class TestLoadMiFIDComplianceConfig:
    """Tests for load_mifid_compliance_config function."""

    def test_load_from_file(self):
        """Test loading configuration from YAML file."""
        config_data = {
            "compliance": {
                "enabled": True,
                "mode": "testing",
                "lei": {
                    "own_lei": "5493001KJTIIGC8Y1R12",
                },
                "nca": {
                    "jurisdiction": "FCA",
                },
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                config = load_mifid_compliance_config(temp_path)
            assert config.enabled is True
            assert config.lei.own_lei == "5493001KJTIIGC8Y1R12"
            assert config.nca.jurisdiction == "FCA"
        finally:
            os.unlink(temp_path)

    def test_load_from_file_without_nesting(self):
        """Test loading configuration without nested key."""
        config_data = {
            "enabled": True,
            "mode": "testing",
            "lei": {
                "cache_ttl_hours": 12,
            },
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                config = load_mifid_compliance_config(temp_path)
            assert config.lei.cache_ttl_hours == 12
        finally:
            os.unlink(temp_path)

    def test_load_missing_file_returns_default(self):
        """Test that missing file returns default configuration."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config = load_mifid_compliance_config("/non/existent/path.yaml")
        assert config.enabled is True
        assert config.mode == ComplianceMode.TESTING

    def test_load_empty_file(self):
        """Test loading empty YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                config = load_mifid_compliance_config(temp_path)
            assert config.enabled is True  # Default value
        finally:
            os.unlink(temp_path)

    def test_load_function_emits_deprecation_warning(self):
        """Test that load function emits deprecation warning."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({}, f)
            temp_path = f.name

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                load_mifid_compliance_config(temp_path)
                # Check that deprecation warning was emitted
                assert any("deprecated" in str(warning.message).lower() for warning in w)
        finally:
            os.unlink(temp_path)


class TestConfigSerialization:
    """Tests for configuration serialization and deserialization."""

    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = MiFIDIIComplianceConfig()
        data = config.model_dump()
        assert isinstance(data, dict)
        assert "enabled" in data
        assert "mode" in data
        assert "lei" in data
        assert "transaction_reporting" in data

    def test_to_json(self):
        """Test converting configuration to JSON."""
        config = MiFIDIIComplianceConfig()
        json_str = config.model_dump_json()
        assert isinstance(json_str, str)
        assert "enabled" in json_str

    def test_from_dict(self):
        """Test creating configuration from dictionary."""
        data = {
            "enabled": False,
            "mode": "disabled",
            "lei": {
                "own_lei": "529900W18LQJJN6SJ336",
            },
        }
        config = MiFIDIIComplianceConfig.model_validate(data)
        assert config.enabled is False
        assert config.mode == ComplianceMode.DISABLED
        assert config.lei.own_lei == "529900W18LQJJN6SJ336"

    def test_roundtrip(self):
        """Test configuration roundtrip (serialize and deserialize)."""
        original = MiFIDIIComplianceConfig(
            enabled=True,
            mode=ComplianceMode.TESTING,
            lei=LEIConfig(own_lei="5493001KJTIIGC8Y1R12"),
            nca=NCANotificationConfig(jurisdiction="AMF"),
        )
        data = original.model_dump()
        restored = MiFIDIIComplianceConfig.model_validate(data)

        assert restored.enabled == original.enabled
        assert restored.mode == original.mode
        assert restored.lei.own_lei == original.lei.own_lei
        assert restored.nca.jurisdiction == original.nca.jurisdiction


class TestDeprecationWarning:
    """Tests for deprecation warning on import."""

    def test_config_module_import_warning(self):
        """Test that importing the config module emits deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Force reload to trigger warning
            import importlib
            import services.archive.mifid_financial_entity.config as cfg_module
            importlib.reload(cfg_module)
            # Check warning was emitted (may have multiple warnings)
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
