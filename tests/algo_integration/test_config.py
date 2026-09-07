# -*- coding: utf-8 -*-
"""
Tests for Algo Integration Configuration Module.

Comprehensive tests for Pydantic configuration models in the INTEGRATION package.
100% coverage for all configuration classes and validation logic.
"""

import pytest
import tempfile
import os
import yaml

from services.algo_integration.config import (
    AlgorithmType,
    ConformanceTestLevel,
    AlgorithmRegistryConfig,
    BestExecutionConfig,
    TCAConfig,
    ConformanceTestingConfig,
    OTRConfig,
    AlgoIntegrationConfig,
    load_algo_integration_config,
)


class TestAlgorithmType:
    """Tests for AlgorithmType enum."""

    def test_execution_value(self):
        """Test EXECUTION enum value."""
        assert AlgorithmType.EXECUTION.value == "execution"

    def test_market_making_value(self):
        """Test MARKET_MAKING enum value."""
        assert AlgorithmType.MARKET_MAKING.value == "market_making"

    def test_arbitrage_value(self):
        """Test ARBITRAGE enum value."""
        assert AlgorithmType.ARBITRAGE.value == "arbitrage"

    def test_trend_following_value(self):
        """Test TREND_FOLLOWING enum value."""
        assert AlgorithmType.TREND_FOLLOWING.value == "trend_following"

    def test_statistical_value(self):
        """Test STATISTICAL enum value."""
        assert AlgorithmType.STATISTICAL.value == "statistical"

    def test_other_value(self):
        """Test OTHER enum value."""
        assert AlgorithmType.OTHER.value == "other"

    def test_all_values(self):
        """Test all enum values are accessible."""
        all_types = [t.value for t in AlgorithmType]
        assert len(all_types) == 6
        assert "execution" in all_types
        assert "market_making" in all_types


class TestConformanceTestLevel:
    """Tests for ConformanceTestLevel enum."""

    def test_basic_value(self):
        """Test BASIC enum value."""
        assert ConformanceTestLevel.BASIC.value == "basic"

    def test_standard_value(self):
        """Test STANDARD enum value."""
        assert ConformanceTestLevel.STANDARD.value == "standard"

    def test_comprehensive_value(self):
        """Test COMPREHENSIVE enum value."""
        assert ConformanceTestLevel.COMPREHENSIVE.value == "comprehensive"

    def test_custom_value(self):
        """Test CUSTOM enum value."""
        assert ConformanceTestLevel.CUSTOM.value == "custom"


class TestAlgorithmRegistryConfig:
    """Tests for AlgorithmRegistryConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AlgorithmRegistryConfig()
        assert config.registry_path == "state/algo_integration/algorithm_registry.json"
        assert config.auto_register is True
        assert config.require_responsible_person is True
        assert config.version_on_modification is True
        assert config.nca_jurisdiction == ""
        assert config.firm_name == ""
        assert config.contact_email == ""
        assert config.contact_phone == ""

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AlgorithmRegistryConfig(
            registry_path="/custom/path/registry.json",
            nca_jurisdiction="FCA",
            firm_name="Test Firm Ltd",
            contact_email="algo@testfirm.com",
            contact_phone="+44123456789",
        )
        assert config.registry_path == "/custom/path/registry.json"
        assert config.nca_jurisdiction == "FCA"
        assert config.firm_name == "Test Firm Ltd"
        assert config.contact_email == "algo@testfirm.com"
        assert config.contact_phone == "+44123456789"

    def test_auto_register_disabled(self):
        """Test disabling auto registration."""
        config = AlgorithmRegistryConfig(auto_register=False)
        assert config.auto_register is False

    def test_require_responsible_person_disabled(self):
        """Test disabling responsible person requirement."""
        config = AlgorithmRegistryConfig(require_responsible_person=False)
        assert config.require_responsible_person is False

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValueError):
            AlgorithmRegistryConfig(unknown_field="value")


class TestBestExecutionConfig:
    """Tests for BestExecutionConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BestExecutionConfig()
        assert config.enabled is True
        assert config.report_frequency == "quarterly"
        assert config.slippage_threshold_bps == 10.0
        assert config.latency_threshold_ms == 100.0
        assert config.venues_to_monitor == []

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BestExecutionConfig(
            enabled=False,
            report_frequency="monthly",
            slippage_threshold_bps=5.0,
            latency_threshold_ms=50.0,
            venues_to_monitor=["LSE", "XETRA", "EURONEXT"],
        )
        assert config.enabled is False
        assert config.report_frequency == "monthly"
        assert config.slippage_threshold_bps == 5.0
        assert config.latency_threshold_ms == 50.0
        assert len(config.venues_to_monitor) == 3

    def test_slippage_threshold_range(self):
        """Test slippage threshold range."""
        config = BestExecutionConfig(slippage_threshold_bps=0.0)
        assert config.slippage_threshold_bps == 0.0

        config = BestExecutionConfig(slippage_threshold_bps=100.0)
        assert config.slippage_threshold_bps == 100.0

    def test_slippage_threshold_invalid(self):
        """Test invalid slippage threshold."""
        with pytest.raises(ValueError):
            BestExecutionConfig(slippage_threshold_bps=-1.0)

        with pytest.raises(ValueError):
            BestExecutionConfig(slippage_threshold_bps=101.0)

    def test_report_frequency_options(self):
        """Test different report frequency options."""
        for freq in ["monthly", "quarterly", "annually"]:
            config = BestExecutionConfig(report_frequency=freq)
            assert config.report_frequency == freq


class TestTCAConfig:
    """Tests for TCAConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TCAConfig()
        assert config.enabled is True
        assert config.benchmark == "VWAP"
        assert config.include_implicit_costs is True
        assert config.report_format == "PDF"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TCAConfig(
            enabled=False,
            benchmark="TWAP",
            include_implicit_costs=False,
            report_format="JSON",
        )
        assert config.enabled is False
        assert config.benchmark == "TWAP"
        assert config.include_implicit_costs is False
        assert config.report_format == "JSON"

    def test_benchmark_options(self):
        """Test different benchmark options."""
        for benchmark in ["VWAP", "TWAP", "Arrival", "Close"]:
            config = TCAConfig(benchmark=benchmark)
            assert config.benchmark == benchmark

    def test_report_format_options(self):
        """Test different report format options."""
        for format in ["PDF", "HTML", "JSON", "CSV"]:
            config = TCAConfig(report_format=format)
            assert config.report_format == format


class TestConformanceTestingConfig:
    """Tests for ConformanceTestingConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ConformanceTestingConfig()
        assert config.enabled is True
        assert config.test_level == ConformanceTestLevel.STANDARD
        assert config.test_environment == "sandbox"
        assert config.require_certification is True
        assert config.certification_validity_days == 365
        assert config.auto_revoke_on_modification is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ConformanceTestingConfig(
            enabled=False,
            test_level=ConformanceTestLevel.COMPREHENSIVE,
            test_environment="uat",
            require_certification=False,
            certification_validity_days=180,
        )
        assert config.enabled is False
        assert config.test_level == ConformanceTestLevel.COMPREHENSIVE
        assert config.test_environment == "uat"
        assert config.require_certification is False
        assert config.certification_validity_days == 180

    def test_certification_validity_days_range(self):
        """Test certification validity days range."""
        config = ConformanceTestingConfig(certification_validity_days=30)
        assert config.certification_validity_days == 30

        config = ConformanceTestingConfig(certification_validity_days=730)
        assert config.certification_validity_days == 730

    def test_certification_validity_days_invalid(self):
        """Test invalid certification validity days."""
        with pytest.raises(ValueError):
            ConformanceTestingConfig(certification_validity_days=29)

        with pytest.raises(ValueError):
            ConformanceTestingConfig(certification_validity_days=731)

    def test_all_test_levels(self):
        """Test all test levels."""
        for level in ConformanceTestLevel:
            config = ConformanceTestingConfig(test_level=level)
            assert config.test_level == level

    def test_test_environment_options(self):
        """Test different test environment options."""
        for env in ["sandbox", "uat", "paper"]:
            config = ConformanceTestingConfig(test_environment=env)
            assert config.test_environment == env


class TestOTRConfig:
    """Tests for OTRConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OTRConfig()
        assert config.enabled is True
        assert config.warning_threshold == 4.0
        assert config.critical_threshold == 8.0
        assert config.window_seconds == 60

    def test_custom_values(self):
        """Test custom configuration values."""
        config = OTRConfig(
            enabled=False,
            warning_threshold=3.0,
            critical_threshold=6.0,
            window_seconds=120,
        )
        assert config.enabled is False
        assert config.warning_threshold == 3.0
        assert config.critical_threshold == 6.0
        assert config.window_seconds == 120

    def test_threshold_validation(self):
        """Test that warning threshold must be less than critical."""
        with pytest.raises(ValueError, match="warning_threshold must be less than critical"):
            OTRConfig(
                warning_threshold=8.0,
                critical_threshold=4.0,
            )

    def test_threshold_validation_equal(self):
        """Test that equal thresholds are invalid."""
        with pytest.raises(ValueError, match="warning_threshold must be less than critical"):
            OTRConfig(
                warning_threshold=5.0,
                critical_threshold=5.0,
            )

    def test_threshold_range(self):
        """Test threshold range."""
        config = OTRConfig(warning_threshold=1.0, critical_threshold=2.0)
        assert config.warning_threshold == 1.0

        config = OTRConfig(warning_threshold=50.0, critical_threshold=100.0)
        assert config.critical_threshold == 100.0

    def test_window_seconds_range(self):
        """Test window seconds range."""
        config = OTRConfig(window_seconds=10)
        assert config.window_seconds == 10

        config = OTRConfig(window_seconds=3600)
        assert config.window_seconds == 3600

    def test_window_seconds_invalid(self):
        """Test invalid window seconds."""
        with pytest.raises(ValueError):
            OTRConfig(window_seconds=9)

        with pytest.raises(ValueError):
            OTRConfig(window_seconds=3601)


class TestAlgoIntegrationConfig:
    """Tests for AlgoIntegrationConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AlgoIntegrationConfig()
        assert config.enabled is False  # Disabled by default for ICT Provider
        assert isinstance(config.algorithm_registry, AlgorithmRegistryConfig)
        assert isinstance(config.best_execution, BestExecutionConfig)
        assert isinstance(config.tca, TCAConfig)
        assert isinstance(config.conformance_testing, ConformanceTestingConfig)
        assert isinstance(config.otr, OTRConfig)

    def test_enabled_for_clients(self):
        """Test enabling for B2B clients."""
        config = AlgoIntegrationConfig(
            enabled=True,
            algorithm_registry=AlgorithmRegistryConfig(
                nca_jurisdiction="FCA",
                firm_name="Client Investment Ltd",
            ),
        )
        assert config.enabled is True
        assert config.algorithm_registry.nca_jurisdiction == "FCA"

    def test_nested_config(self):
        """Test nested configuration objects."""
        config = AlgoIntegrationConfig(
            enabled=True,
            algorithm_registry=AlgorithmRegistryConfig(
                registry_path="/custom/registry.json",
                firm_name="Test Firm",
            ),
            best_execution=BestExecutionConfig(
                report_frequency="monthly",
                venues_to_monitor=["LSE"],
            ),
            tca=TCAConfig(
                benchmark="TWAP",
            ),
            conformance_testing=ConformanceTestingConfig(
                test_level=ConformanceTestLevel.COMPREHENSIVE,
            ),
            otr=OTRConfig(
                warning_threshold=3.0,
                critical_threshold=5.0,
            ),
        )
        assert config.algorithm_registry.firm_name == "Test Firm"
        assert config.best_execution.report_frequency == "monthly"
        assert config.tca.benchmark == "TWAP"
        assert config.conformance_testing.test_level == ConformanceTestLevel.COMPREHENSIVE
        assert config.otr.warning_threshold == 3.0

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed in top-level config."""
        config = AlgoIntegrationConfig(custom_field="custom_value")
        # Should not raise - extra fields allowed


class TestLoadAlgoIntegrationConfig:
    """Tests for load_algo_integration_config function."""

    def test_load_from_file(self):
        """Test loading configuration from YAML file."""
        config_data = {
            "algo_integration": {
                "enabled": True,
                "algorithm_registry": {
                    "nca_jurisdiction": "FCA",
                    "firm_name": "Test Firm",
                },
                "best_execution": {
                    "report_frequency": "monthly",
                },
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_algo_integration_config(temp_path)
            assert config.enabled is True
            assert config.algorithm_registry.nca_jurisdiction == "FCA"
            assert config.best_execution.report_frequency == "monthly"
        finally:
            os.unlink(temp_path)

    def test_load_from_file_without_nesting(self):
        """Test loading configuration without nested key."""
        config_data = {
            "enabled": True,
            "algorithm_registry": {
                "firm_name": "Direct Config Ltd",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_algo_integration_config(temp_path)
            assert config.algorithm_registry.firm_name == "Direct Config Ltd"
        finally:
            os.unlink(temp_path)

    def test_load_missing_file_returns_default(self):
        """Test that missing file returns default configuration."""
        config = load_algo_integration_config("/non/existent/path.yaml")
        assert config.enabled is False  # Default is False

    def test_load_empty_file(self):
        """Test loading empty YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            config = load_algo_integration_config(temp_path)
            assert config.enabled is False  # Default value
        finally:
            os.unlink(temp_path)

    def test_load_full_config(self):
        """Test loading full configuration with all nested objects."""
        config_data = {
            "algo_integration": {
                "enabled": True,
                "algorithm_registry": {
                    "nca_jurisdiction": "BaFin",
                    "firm_name": "German Investment GmbH",
                    "contact_email": "algo@german-invest.de",
                },
                "best_execution": {
                    "slippage_threshold_bps": 5.0,
                    "venues_to_monitor": ["XETRA", "EURONEXT"],
                },
                "tca": {
                    "benchmark": "Arrival",
                    "report_format": "HTML",
                },
                "conformance_testing": {
                    "test_level": "comprehensive",
                    "certification_validity_days": 180,
                },
                "otr": {
                    "warning_threshold": 3.0,
                    "critical_threshold": 6.0,
                },
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_algo_integration_config(temp_path)
            assert config.enabled is True
            assert config.algorithm_registry.nca_jurisdiction == "BaFin"
            assert config.best_execution.slippage_threshold_bps == 5.0
            assert "XETRA" in config.best_execution.venues_to_monitor
            assert config.tca.benchmark == "Arrival"
            assert config.conformance_testing.test_level == ConformanceTestLevel.COMPREHENSIVE
            assert config.otr.warning_threshold == 3.0
        finally:
            os.unlink(temp_path)


class TestConfigSerialization:
    """Tests for configuration serialization and deserialization."""

    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = AlgoIntegrationConfig()
        data = config.model_dump()
        assert isinstance(data, dict)
        assert "enabled" in data
        assert "algorithm_registry" in data
        assert "best_execution" in data

    def test_to_json(self):
        """Test converting configuration to JSON."""
        config = AlgoIntegrationConfig()
        json_str = config.model_dump_json()
        assert isinstance(json_str, str)
        assert "enabled" in json_str

    def test_from_dict(self):
        """Test creating configuration from dictionary."""
        data = {
            "enabled": True,
            "algorithm_registry": {
                "nca_jurisdiction": "AMF",
            },
        }
        config = AlgoIntegrationConfig.model_validate(data)
        assert config.enabled is True
        assert config.algorithm_registry.nca_jurisdiction == "AMF"

    def test_roundtrip(self):
        """Test configuration roundtrip (serialize and deserialize)."""
        original = AlgoIntegrationConfig(
            enabled=True,
            algorithm_registry=AlgorithmRegistryConfig(nca_jurisdiction="FCA"),
            best_execution=BestExecutionConfig(report_frequency="monthly"),
        )
        data = original.model_dump()
        restored = AlgoIntegrationConfig.model_validate(data)

        assert restored.enabled == original.enabled
        assert (
            restored.algorithm_registry.nca_jurisdiction
            == original.algorithm_registry.nca_jurisdiction
        )
        assert restored.best_execution.report_frequency == original.best_execution.report_frequency
