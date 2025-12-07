# -*- coding: utf-8 -*-
"""
Tests for MiFID II Compliance Configuration Module.

Comprehensive tests for Pydantic configuration models.
"""

import pytest
import tempfile
import os
import yaml

from services.compliance.config import (
    ComplianceMode,
    LEIConfig,
    ClockSyncComplianceConfig,
    AlgorithmRegistryConfig,
    PreTradeControlsConfig,
    MiFIDIIComplianceConfig,
    load_compliance_config,
)


class TestComplianceMode:
    """Tests for ComplianceMode enum."""

    def test_production_value(self):
        assert ComplianceMode.PRODUCTION.value == "production"

    def test_testing_value(self):
        assert ComplianceMode.TESTING.value == "testing"

    def test_disabled_value(self):
        assert ComplianceMode.DISABLED.value == "disabled"


class TestLEIConfig:
    """Tests for LEIConfig model."""

    def test_default_values(self):
        config = LEIConfig()
        assert config.own_lei == ""
        assert config.gleif_api_url == "https://api.gleif.org/api/v1"
        assert config.cache_ttl_hours == 24
        assert config.verify_before_trade is True
        assert config.renewal_warning_days == 30

    def test_custom_values(self):
        config = LEIConfig(
            own_lei="5493001KJTIIGC8Y1R12",
            cache_ttl_hours=12,
            verify_before_trade=False,
        )
        assert config.own_lei == "5493001KJTIIGC8Y1R12"
        assert config.cache_ttl_hours == 12
        assert config.verify_before_trade is False

    def test_cache_ttl_min_max(self):
        # Should accept valid range
        config = LEIConfig(cache_ttl_hours=1)
        assert config.cache_ttl_hours == 1

        config = LEIConfig(cache_ttl_hours=168)
        assert config.cache_ttl_hours == 168

        # Should reject invalid values
        with pytest.raises(ValueError):
            LEIConfig(cache_ttl_hours=0)

        with pytest.raises(ValueError):
            LEIConfig(cache_ttl_hours=200)

    def test_renewal_warning_days_range(self):
        config = LEIConfig(renewal_warning_days=7)
        assert config.renewal_warning_days == 7

        config = LEIConfig(renewal_warning_days=90)
        assert config.renewal_warning_days == 90

        with pytest.raises(ValueError):
            LEIConfig(renewal_warning_days=5)


class TestClockSyncComplianceConfig:
    """Tests for ClockSyncComplianceConfig model."""

    def test_default_values(self):
        config = ClockSyncComplianceConfig()
        assert len(config.ntp_servers) == 4
        assert config.max_offset_ms == 100.0
        assert config.sync_interval_seconds == 60
        assert config.warning_threshold_ms == 50.0
        assert config.critical_threshold_ms == 100.0
        assert config.kill_switch_threshold_ms == 1000.0

    def test_custom_ntp_servers(self):
        servers = ["custom.ntp1.com", "custom.ntp2.com"]
        config = ClockSyncComplianceConfig(ntp_servers=servers)
        assert config.ntp_servers == servers

    def test_threshold_validation_warning_less_than_critical(self):
        with pytest.raises(ValueError, match="warning_threshold_ms must be less than critical"):
            ClockSyncComplianceConfig(
                warning_threshold_ms=100.0,
                critical_threshold_ms=50.0,
            )

    def test_threshold_validation_critical_less_than_kill_switch(self):
        with pytest.raises(ValueError, match="critical_threshold_ms must be less than kill_switch"):
            ClockSyncComplianceConfig(
                warning_threshold_ms=50.0,
                critical_threshold_ms=1500.0,
                kill_switch_threshold_ms=1000.0,
            )

    def test_valid_thresholds(self):
        config = ClockSyncComplianceConfig(
            warning_threshold_ms=25.0,
            critical_threshold_ms=50.0,
            kill_switch_threshold_ms=500.0,
        )
        assert config.warning_threshold_ms == 25.0
        assert config.critical_threshold_ms == 50.0
        assert config.kill_switch_threshold_ms == 500.0

    def test_sync_interval_range(self):
        config = ClockSyncComplianceConfig(sync_interval_seconds=10)
        assert config.sync_interval_seconds == 10

        config = ClockSyncComplianceConfig(sync_interval_seconds=3600)
        assert config.sync_interval_seconds == 3600

        with pytest.raises(ValueError):
            ClockSyncComplianceConfig(sync_interval_seconds=5)

    def test_stratum_max_range(self):
        config = ClockSyncComplianceConfig(stratum_max=1)
        assert config.stratum_max == 1

        config = ClockSyncComplianceConfig(stratum_max=15)
        assert config.stratum_max == 15


class TestAlgorithmRegistryConfig:
    """Tests for AlgorithmRegistryConfig model."""

    def test_default_values(self):
        config = AlgorithmRegistryConfig()
        assert config.registry_path == "state/compliance/algorithm_registry.json"
        assert config.auto_register is True
        assert config.require_responsible_person is True
        assert config.version_on_modification is True

    def test_custom_values(self):
        config = AlgorithmRegistryConfig(
            registry_path="/custom/path/registry.json",
            nca_jurisdiction="FCA",
            firm_name="Test Firm Ltd",
        )
        assert config.registry_path == "/custom/path/registry.json"
        assert config.nca_jurisdiction == "FCA"
        assert config.firm_name == "Test Firm Ltd"


class TestPreTradeControlsConfig:
    """Tests for PreTradeControlsConfig model."""

    def test_default_values(self):
        config = PreTradeControlsConfig()
        assert config.price_collar_pct == 5.0
        assert config.max_order_value_eur == 1_000_000.0
        assert config.max_messages_per_second == 100

    def test_price_collar_range(self):
        config = PreTradeControlsConfig(price_collar_pct=0.1)
        assert config.price_collar_pct == 0.1

        config = PreTradeControlsConfig(price_collar_pct=50.0)
        assert config.price_collar_pct == 50.0

        with pytest.raises(ValueError):
            PreTradeControlsConfig(price_collar_pct=0.05)

    def test_fat_finger_values(self):
        config = PreTradeControlsConfig(
            fat_finger_price_deviation_pct=15.0,
            fat_finger_volume_multiplier=20.0,
        )
        assert config.fat_finger_price_deviation_pct == 15.0
        assert config.fat_finger_volume_multiplier == 20.0


class TestMiFIDIIComplianceConfig:
    """Tests for MiFIDIIComplianceConfig model."""

    def test_default_values(self):
        config = MiFIDIIComplianceConfig()
        assert config.enabled is True
        assert config.mode == ComplianceMode.TESTING
        assert isinstance(config.lei, LEIConfig)
        assert isinstance(config.clock, ClockSyncComplianceConfig)
        assert isinstance(config.algorithm_registry, AlgorithmRegistryConfig)

    def test_production_mode_requires_lei(self):
        with pytest.raises(ValueError, match="LEI is required for production"):
            MiFIDIIComplianceConfig(
                mode=ComplianceMode.PRODUCTION,
                lei=LEIConfig(own_lei=""),
                algorithm_registry=AlgorithmRegistryConfig(nca_jurisdiction="FCA"),
            )

    def test_production_mode_requires_nca(self):
        with pytest.raises(ValueError, match="NCA jurisdiction is required"):
            MiFIDIIComplianceConfig(
                mode=ComplianceMode.PRODUCTION,
                lei=LEIConfig(own_lei="5493001KJTIIGC8Y1R12"),
                algorithm_registry=AlgorithmRegistryConfig(nca_jurisdiction=""),
            )

    def test_production_mode_valid(self):
        config = MiFIDIIComplianceConfig(
            mode=ComplianceMode.PRODUCTION,
            lei=LEIConfig(own_lei="5493001KJTIIGC8Y1R12"),
            algorithm_registry=AlgorithmRegistryConfig(nca_jurisdiction="FCA"),
        )
        assert config.mode == ComplianceMode.PRODUCTION

    def test_testing_mode_no_requirements(self):
        config = MiFIDIIComplianceConfig(
            mode=ComplianceMode.TESTING,
            lei=LEIConfig(own_lei=""),
        )
        assert config.mode == ComplianceMode.TESTING

    def test_disabled_mode(self):
        config = MiFIDIIComplianceConfig(
            enabled=False,
            mode=ComplianceMode.DISABLED,
        )
        assert config.enabled is False

    def test_nested_config(self):
        config = MiFIDIIComplianceConfig(
            lei=LEIConfig(
                own_lei="5493001KJTIIGC8Y1R12",
                cache_ttl_hours=12,
            ),
            clock=ClockSyncComplianceConfig(
                max_offset_ms=50.0,
                sync_interval_seconds=120,
            ),
        )
        assert config.lei.own_lei == "5493001KJTIIGC8Y1R12"
        assert config.clock.max_offset_ms == 50.0


class TestLoadComplianceConfig:
    """Tests for load_compliance_config function."""

    def test_load_from_file(self):
        config_data = {
            "compliance": {
                "enabled": True,
                "mode": "testing",
                "lei": {
                    "own_lei": "5493001KJTIIGC8Y1R12",
                },
                "clock": {
                    "max_offset_ms": 50.0,
                },
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_compliance_config(temp_path)
            assert config.enabled is True
            assert config.lei.own_lei == "5493001KJTIIGC8Y1R12"
            assert config.clock.max_offset_ms == 50.0
        finally:
            os.unlink(temp_path)

    def test_load_from_file_without_nesting(self):
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
            config = load_compliance_config(temp_path)
            assert config.lei.cache_ttl_hours == 12
        finally:
            os.unlink(temp_path)

    def test_load_missing_file_returns_default(self):
        config = load_compliance_config("/non/existent/path.yaml")
        assert config.enabled is True
        assert config.mode == ComplianceMode.TESTING

    def test_load_empty_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            config = load_compliance_config(temp_path)
            assert config.enabled is True  # Default value
        finally:
            os.unlink(temp_path)
