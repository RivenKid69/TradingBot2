# -*- coding: utf-8 -*-
"""
Tests for Core Risk Controls Configuration Module.

Comprehensive tests for Pydantic configuration models in the CORE package.
100% coverage for all configuration classes and validation logic.
"""

import pytest
import tempfile
import os
import yaml

from services.core.risk_controls.config import (
    ControlsMode,
    TimeSyncConfig,
    PreTradeControlsConfig,
    AuditConfig,
    KillSwitchConfig,
    RiskControlsConfig,
    load_risk_controls_config,
)


class TestControlsMode:
    """Tests for ControlsMode enum."""

    def test_production_value(self):
        """Test PRODUCTION enum value."""
        assert ControlsMode.PRODUCTION.value == "production"

    def test_testing_value(self):
        """Test TESTING enum value."""
        assert ControlsMode.TESTING.value == "testing"

    def test_disabled_value(self):
        """Test DISABLED enum value."""
        assert ControlsMode.DISABLED.value == "disabled"

    def test_enum_from_string(self):
        """Test creating enum from string."""
        assert ControlsMode("production") == ControlsMode.PRODUCTION
        assert ControlsMode("testing") == ControlsMode.TESTING
        assert ControlsMode("disabled") == ControlsMode.DISABLED


class TestTimeSyncConfig:
    """Tests for TimeSyncConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TimeSyncConfig()
        assert len(config.ntp_servers) == 4
        assert "time.google.com" in config.ntp_servers
        assert config.max_offset_ms == 100.0
        assert config.sync_interval_seconds == 60
        assert config.warning_threshold_ms == 50.0
        assert config.critical_threshold_ms == 100.0
        assert config.kill_switch_threshold_ms == 1000.0
        assert config.timestamp_precision == "nanosecond"
        assert config.fallback_to_system is True
        assert config.stratum_max == 3

    def test_custom_ntp_servers(self):
        """Test custom NTP servers configuration."""
        servers = ["custom.ntp1.com", "custom.ntp2.com"]
        config = TimeSyncConfig(ntp_servers=servers)
        assert config.ntp_servers == servers

    def test_threshold_validation_warning_less_than_critical(self):
        """Test that warning threshold must be less than critical."""
        with pytest.raises(ValueError, match="warning_threshold_ms must be less than critical"):
            TimeSyncConfig(
                warning_threshold_ms=100.0,
                critical_threshold_ms=50.0,
            )

    def test_threshold_validation_critical_less_than_kill_switch(self):
        """Test that critical threshold must be less than kill switch."""
        with pytest.raises(ValueError, match="critical_threshold_ms must be less than kill_switch"):
            TimeSyncConfig(
                warning_threshold_ms=50.0,
                critical_threshold_ms=1500.0,
                kill_switch_threshold_ms=1000.0,
            )

    def test_valid_thresholds(self):
        """Test valid threshold configuration."""
        config = TimeSyncConfig(
            warning_threshold_ms=25.0,
            critical_threshold_ms=50.0,
            kill_switch_threshold_ms=500.0,
        )
        assert config.warning_threshold_ms == 25.0
        assert config.critical_threshold_ms == 50.0
        assert config.kill_switch_threshold_ms == 500.0

    def test_sync_interval_range_min(self):
        """Test minimum sync interval."""
        config = TimeSyncConfig(sync_interval_seconds=10)
        assert config.sync_interval_seconds == 10

    def test_sync_interval_range_max(self):
        """Test maximum sync interval."""
        config = TimeSyncConfig(sync_interval_seconds=3600)
        assert config.sync_interval_seconds == 3600

    def test_sync_interval_below_min(self):
        """Test sync interval below minimum."""
        with pytest.raises(ValueError):
            TimeSyncConfig(sync_interval_seconds=5)

    def test_sync_interval_above_max(self):
        """Test sync interval above maximum."""
        with pytest.raises(ValueError):
            TimeSyncConfig(sync_interval_seconds=4000)

    def test_stratum_max_range(self):
        """Test stratum max range."""
        config = TimeSyncConfig(stratum_max=1)
        assert config.stratum_max == 1

        config = TimeSyncConfig(stratum_max=15)
        assert config.stratum_max == 15

    def test_stratum_max_invalid(self):
        """Test invalid stratum max values."""
        with pytest.raises(ValueError):
            TimeSyncConfig(stratum_max=0)

        with pytest.raises(ValueError):
            TimeSyncConfig(stratum_max=16)

    def test_max_offset_ms_range(self):
        """Test max offset ms range."""
        config = TimeSyncConfig(max_offset_ms=0.001)
        assert config.max_offset_ms == 0.001

        config = TimeSyncConfig(max_offset_ms=1000.0)
        assert config.max_offset_ms == 1000.0

    def test_max_offset_ms_invalid(self):
        """Test invalid max offset ms."""
        with pytest.raises(ValueError):
            TimeSyncConfig(max_offset_ms=0.0)

    def test_timestamp_precision_options(self):
        """Test different timestamp precision options."""
        for precision in ["nanosecond", "microsecond", "millisecond"]:
            config = TimeSyncConfig(timestamp_precision=precision)
            assert config.timestamp_precision == precision

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValueError):
            TimeSyncConfig(unknown_field="value")


class TestPreTradeControlsConfig:
    """Tests for PreTradeControlsConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PreTradeControlsConfig()
        assert config.price_collar_pct == 5.0
        assert config.max_order_value_eur == 1_000_000.0
        assert config.max_order_volume == 10_000.0
        assert config.max_messages_per_second == 100
        assert config.fat_finger_price_deviation_pct == 10.0
        assert config.fat_finger_volume_multiplier == 10.0

    def test_price_collar_range_min(self):
        """Test minimum price collar."""
        config = PreTradeControlsConfig(price_collar_pct=0.1)
        assert config.price_collar_pct == 0.1

    def test_price_collar_range_max(self):
        """Test maximum price collar."""
        config = PreTradeControlsConfig(price_collar_pct=50.0)
        assert config.price_collar_pct == 50.0

    def test_price_collar_invalid(self):
        """Test invalid price collar values."""
        with pytest.raises(ValueError):
            PreTradeControlsConfig(price_collar_pct=0.05)

        with pytest.raises(ValueError):
            PreTradeControlsConfig(price_collar_pct=51.0)

    def test_fat_finger_values(self):
        """Test fat finger protection values."""
        config = PreTradeControlsConfig(
            fat_finger_price_deviation_pct=15.0,
            fat_finger_volume_multiplier=20.0,
        )
        assert config.fat_finger_price_deviation_pct == 15.0
        assert config.fat_finger_volume_multiplier == 20.0

    def test_fat_finger_deviation_range(self):
        """Test fat finger deviation range."""
        config = PreTradeControlsConfig(fat_finger_price_deviation_pct=1.0)
        assert config.fat_finger_price_deviation_pct == 1.0

        config = PreTradeControlsConfig(fat_finger_price_deviation_pct=100.0)
        assert config.fat_finger_price_deviation_pct == 100.0

    def test_fat_finger_volume_multiplier_range(self):
        """Test fat finger volume multiplier range."""
        config = PreTradeControlsConfig(fat_finger_volume_multiplier=2.0)
        assert config.fat_finger_volume_multiplier == 2.0

        config = PreTradeControlsConfig(fat_finger_volume_multiplier=100.0)
        assert config.fat_finger_volume_multiplier == 100.0

    def test_max_messages_per_second_range(self):
        """Test max messages per second range."""
        config = PreTradeControlsConfig(max_messages_per_second=1)
        assert config.max_messages_per_second == 1

        config = PreTradeControlsConfig(max_messages_per_second=10_000)
        assert config.max_messages_per_second == 10_000

    def test_max_messages_per_second_invalid(self):
        """Test invalid max messages per second."""
        with pytest.raises(ValueError):
            PreTradeControlsConfig(max_messages_per_second=0)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValueError):
            PreTradeControlsConfig(unknown_field="value")


class TestAuditConfig:
    """Tests for AuditConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AuditConfig()
        assert config.log_path == "logs/risk_controls/audit.log"
        assert config.retention_years == 5
        assert config.max_file_size_mb == 100
        assert config.compress_archived is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AuditConfig(
            log_path="/custom/path/audit.log",
            retention_years=7,
            max_file_size_mb=500,
            compress_archived=False,
        )
        assert config.log_path == "/custom/path/audit.log"
        assert config.retention_years == 7
        assert config.max_file_size_mb == 500
        assert config.compress_archived is False

    def test_retention_years_range(self):
        """Test retention years range."""
        config = AuditConfig(retention_years=1)
        assert config.retention_years == 1

        config = AuditConfig(retention_years=10)
        assert config.retention_years == 10

    def test_retention_years_invalid(self):
        """Test invalid retention years."""
        with pytest.raises(ValueError):
            AuditConfig(retention_years=0)

        with pytest.raises(ValueError):
            AuditConfig(retention_years=11)

    def test_max_file_size_mb_range(self):
        """Test max file size MB range."""
        config = AuditConfig(max_file_size_mb=10)
        assert config.max_file_size_mb == 10

        config = AuditConfig(max_file_size_mb=1000)
        assert config.max_file_size_mb == 1000


class TestKillSwitchConfig:
    """Tests for KillSwitchConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = KillSwitchConfig()
        assert config.enabled is True
        assert config.max_daily_loss_pct == 5.0
        assert config.max_position_value_eur == 10_000_000.0
        assert config.auto_reset_enabled is False
        assert config.cool_down_minutes == 60
        assert config.notification_emails == []
        assert config.require_manual_reset is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = KillSwitchConfig(
            enabled=False,
            max_daily_loss_pct=2.5,
            notification_emails=["alerts@example.com", "risk@example.com"],
        )
        assert config.enabled is False
        assert config.max_daily_loss_pct == 2.5
        assert len(config.notification_emails) == 2

    def test_max_daily_loss_pct_range(self):
        """Test max daily loss percentage range."""
        config = KillSwitchConfig(max_daily_loss_pct=0.1)
        assert config.max_daily_loss_pct == 0.1

        config = KillSwitchConfig(max_daily_loss_pct=100.0)
        assert config.max_daily_loss_pct == 100.0

    def test_cool_down_minutes_range(self):
        """Test cool down minutes range."""
        config = KillSwitchConfig(cool_down_minutes=5)
        assert config.cool_down_minutes == 5

        config = KillSwitchConfig(cool_down_minutes=1440)
        assert config.cool_down_minutes == 1440


class TestRiskControlsConfig:
    """Tests for RiskControlsConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RiskControlsConfig()
        assert config.enabled is True
        assert config.mode == ControlsMode.TESTING
        assert isinstance(config.time_sync, TimeSyncConfig)
        assert isinstance(config.pre_trade, PreTradeControlsConfig)
        assert isinstance(config.audit, AuditConfig)
        assert isinstance(config.kill_switch, KillSwitchConfig)

    def test_production_mode_requires_kill_switch(self):
        """Test that production mode requires kill switch."""
        with pytest.raises(ValueError, match="Kill switch must be enabled in production"):
            RiskControlsConfig(
                mode=ControlsMode.PRODUCTION,
                kill_switch=KillSwitchConfig(enabled=False),
            )

    def test_production_mode_valid(self):
        """Test valid production mode configuration."""
        config = RiskControlsConfig(
            mode=ControlsMode.PRODUCTION,
            kill_switch=KillSwitchConfig(enabled=True),
        )
        assert config.mode == ControlsMode.PRODUCTION

    def test_testing_mode_no_requirements(self):
        """Test that testing mode has no special requirements."""
        config = RiskControlsConfig(
            mode=ControlsMode.TESTING,
            kill_switch=KillSwitchConfig(enabled=False),
        )
        assert config.mode == ControlsMode.TESTING

    def test_disabled_mode(self):
        """Test disabled mode configuration."""
        config = RiskControlsConfig(
            enabled=False,
            mode=ControlsMode.DISABLED,
        )
        assert config.enabled is False
        assert config.mode == ControlsMode.DISABLED

    def test_nested_config(self):
        """Test nested configuration objects."""
        config = RiskControlsConfig(
            time_sync=TimeSyncConfig(
                max_offset_ms=50.0,
                sync_interval_seconds=120,
            ),
            pre_trade=PreTradeControlsConfig(
                price_collar_pct=3.0,
                max_order_value_eur=500_000.0,
            ),
        )
        assert config.time_sync.max_offset_ms == 50.0
        assert config.pre_trade.price_collar_pct == 3.0

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed in top-level config."""
        config = RiskControlsConfig(custom_field="custom_value")
        # Should not raise - extra fields allowed


class TestLoadRiskControlsConfig:
    """Tests for load_risk_controls_config function."""

    def test_load_from_file(self):
        """Test loading configuration from YAML file."""
        config_data = {
            "risk_controls": {
                "enabled": True,
                "mode": "testing",
                "time_sync": {
                    "max_offset_ms": 50.0,
                },
                "pre_trade": {
                    "price_collar_pct": 3.0,
                },
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_risk_controls_config(temp_path)
            assert config.enabled is True
            assert config.time_sync.max_offset_ms == 50.0
            assert config.pre_trade.price_collar_pct == 3.0
        finally:
            os.unlink(temp_path)

    def test_load_from_file_without_nesting(self):
        """Test loading configuration without nested key."""
        config_data = {
            "enabled": True,
            "mode": "testing",
            "time_sync": {
                "sync_interval_seconds": 120,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_risk_controls_config(temp_path)
            assert config.time_sync.sync_interval_seconds == 120
        finally:
            os.unlink(temp_path)

    def test_load_missing_file_returns_default(self):
        """Test that missing file returns default configuration."""
        config = load_risk_controls_config("/non/existent/path.yaml")
        assert config.enabled is True
        assert config.mode == ControlsMode.TESTING

    def test_load_empty_file(self):
        """Test loading empty YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            config = load_risk_controls_config(temp_path)
            assert config.enabled is True  # Default value
        finally:
            os.unlink(temp_path)

    def test_load_full_config(self):
        """Test loading full configuration with all nested objects."""
        config_data = {
            "risk_controls": {
                "enabled": True,
                "mode": "production",
                "time_sync": {
                    "ntp_servers": ["ntp.example.com"],
                    "max_offset_ms": 75.0,
                    "warning_threshold_ms": 25.0,
                    "critical_threshold_ms": 50.0,
                    "kill_switch_threshold_ms": 500.0,
                },
                "pre_trade": {
                    "price_collar_pct": 2.5,
                    "max_order_value_eur": 250000.0,
                },
                "audit": {
                    "retention_years": 7,
                },
                "kill_switch": {
                    "enabled": True,
                    "max_daily_loss_pct": 3.0,
                },
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_risk_controls_config(temp_path)
            assert config.mode == ControlsMode.PRODUCTION
            assert config.time_sync.ntp_servers == ["ntp.example.com"]
            assert config.time_sync.max_offset_ms == 75.0
            assert config.pre_trade.price_collar_pct == 2.5
            assert config.audit.retention_years == 7
            assert config.kill_switch.max_daily_loss_pct == 3.0
        finally:
            os.unlink(temp_path)


class TestConfigSerialization:
    """Tests for configuration serialization and deserialization."""

    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = RiskControlsConfig()
        data = config.model_dump()
        assert isinstance(data, dict)
        assert "enabled" in data
        assert "mode" in data
        assert "time_sync" in data
        assert "pre_trade" in data

    def test_to_json(self):
        """Test converting configuration to JSON."""
        config = RiskControlsConfig()
        json_str = config.model_dump_json()
        assert isinstance(json_str, str)
        assert "enabled" in json_str

    def test_from_dict(self):
        """Test creating configuration from dictionary."""
        data = {
            "enabled": False,
            "mode": "disabled",
            "time_sync": {
                "max_offset_ms": 200.0,
            },
        }
        config = RiskControlsConfig.model_validate(data)
        assert config.enabled is False
        assert config.mode == ControlsMode.DISABLED
        assert config.time_sync.max_offset_ms == 200.0

    def test_roundtrip(self):
        """Test configuration roundtrip (serialize and deserialize)."""
        original = RiskControlsConfig(
            enabled=True,
            mode=ControlsMode.TESTING,
            time_sync=TimeSyncConfig(max_offset_ms=75.0),
            pre_trade=PreTradeControlsConfig(price_collar_pct=3.0),
        )
        data = original.model_dump()
        restored = RiskControlsConfig.model_validate(data)

        assert restored.enabled == original.enabled
        assert restored.mode == original.mode
        assert restored.time_sync.max_offset_ms == original.time_sync.max_offset_ms
        assert restored.pre_trade.price_collar_pct == original.pre_trade.price_collar_pct
