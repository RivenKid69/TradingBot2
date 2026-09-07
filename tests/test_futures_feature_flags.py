# -*- coding: utf-8 -*-
"""
Test suite for futures feature flags system.

Tests cover:
1. Feature flag configuration and validation
2. Rollout stage mechanics (disabled, shadow, canary, production)
3. Global kill switch functionality
4. Canary percentage evaluation
5. Symbol and account filtering
6. Feature flag decorator
7. YAML configuration loading
8. Thread safety for concurrent access

Target: 100% coverage for services/futures_feature_flags.py
"""

import os
import random
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict

import pytest


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_flags_yaml():
    """Sample feature flags YAML content."""
    return """
global_kill_switch: false
environment: staging

features:
  perpetual_trading:
    stage: production
    canary_percentage: 100

  quarterly_trading:
    stage: canary
    canary_percentage: 25
    allowed_symbols:
      - BTCUSDT_QUARTERLY
      - ETHUSDT_QUARTERLY

  index_futures:
    stage: shadow
    allowed_symbols:
      - ES
      - NQ

  commodity_futures:
    stage: disabled

  l3_execution:
    stage: canary
    canary_percentage: 10
    allowed_symbols:
      - BTCUSDT
      - ETHUSDT

  leverage_action_space:
    stage: canary
    canary_percentage: 50

  futures_env_wrapper:
    stage: production
"""


@pytest.fixture
def flags_yaml_path(tmp_path, sample_flags_yaml):
    """Create temporary YAML file with sample flags."""
    yaml_path = tmp_path / "test_feature_flags.yaml"
    yaml_path.write_text(sample_flags_yaml)
    return yaml_path


@pytest.fixture(autouse=True)
def reset_global_flags_fixture():
    """Reset global flags before and after each test."""
    from services.futures_feature_flags import reset_global_flags

    reset_global_flags()
    yield
    reset_global_flags()


# =============================================================================
# TEST IMPORTS
# =============================================================================


class TestImports:
    """Test module imports."""

    def test_import_all_components(self):
        """Test that all components can be imported."""
        from services.futures_feature_flags import (
            RolloutStage,
            FuturesFeature,
            FeatureConfig,
            FuturesFeatureFlags,
            get_global_flags,
            set_global_flags,
            reset_global_flags,
            init_feature_flags,
            feature_flag,
            require_feature,
            enable_all_for_testing,
            create_minimal_crypto_flags,
            create_minimal_cme_flags,
        )

        assert RolloutStage is not None
        assert FuturesFeature is not None
        assert FeatureConfig is not None
        assert FuturesFeatureFlags is not None
        assert get_global_flags is not None
        assert set_global_flags is not None
        assert reset_global_flags is not None
        assert init_feature_flags is not None
        assert feature_flag is not None
        assert require_feature is not None
        assert enable_all_for_testing is not None
        assert create_minimal_crypto_flags is not None
        assert create_minimal_cme_flags is not None


# =============================================================================
# TEST ROLLOUT STAGE ENUM
# =============================================================================


class TestRolloutStage:
    """Test RolloutStage enum."""

    def test_rollout_stage_values(self):
        """Test RolloutStage enum values."""
        from services.futures_feature_flags import RolloutStage

        assert RolloutStage.DISABLED.value == "disabled"
        assert RolloutStage.SHADOW.value == "shadow"
        assert RolloutStage.CANARY.value == "canary"
        assert RolloutStage.PRODUCTION.value == "production"

    def test_rollout_stage_comparison(self):
        """Test RolloutStage comparisons."""
        from services.futures_feature_flags import RolloutStage

        assert RolloutStage.DISABLED != RolloutStage.PRODUCTION
        assert RolloutStage.CANARY != RolloutStage.SHADOW

    def test_rollout_stage_from_string(self):
        """Test RolloutStage from string."""
        from services.futures_feature_flags import RolloutStage

        assert RolloutStage("disabled") == RolloutStage.DISABLED
        assert RolloutStage("shadow") == RolloutStage.SHADOW
        assert RolloutStage("canary") == RolloutStage.CANARY
        assert RolloutStage("production") == RolloutStage.PRODUCTION


# =============================================================================
# TEST FUTURES FEATURE ENUM
# =============================================================================


class TestFuturesFeature:
    """Test FuturesFeature enum."""

    def test_core_trading_features(self):
        """Test core trading feature enums."""
        from services.futures_feature_flags import FuturesFeature

        assert FuturesFeature.PERPETUAL_TRADING.value == "perpetual_trading"
        assert FuturesFeature.QUARTERLY_TRADING.value == "quarterly_trading"
        assert FuturesFeature.INDEX_FUTURES.value == "index_futures"
        assert FuturesFeature.COMMODITY_FUTURES.value == "commodity_futures"

    def test_margin_features(self):
        """Test margin feature enums."""
        from services.futures_feature_flags import FuturesFeature

        assert FuturesFeature.CROSS_MARGIN.value == "cross_margin"
        assert FuturesFeature.ISOLATED_MARGIN.value == "isolated_margin"
        assert FuturesFeature.LIQUIDATION_SIMULATION.value == "liquidation_simulation"
        assert FuturesFeature.SPAN_MARGIN.value == "span_margin"

    def test_funding_features(self):
        """Test funding feature enums."""
        from services.futures_feature_flags import FuturesFeature

        assert FuturesFeature.FUNDING_RATE_TRACKING.value == "funding_rate_tracking"
        assert FuturesFeature.FUNDING_IN_REWARD.value == "funding_in_reward"

    def test_execution_features(self):
        """Test execution feature enums."""
        from services.futures_feature_flags import FuturesFeature

        assert FuturesFeature.L2_EXECUTION.value == "l2_execution"
        assert FuturesFeature.L3_EXECUTION.value == "l3_execution"

    def test_training_features(self):
        """Test training feature enums."""
        from services.futures_feature_flags import FuturesFeature

        assert FuturesFeature.FUTURES_ENV_WRAPPER.value == "futures_env_wrapper"
        assert FuturesFeature.LEVERAGE_ACTION_SPACE.value == "leverage_action_space"


# =============================================================================
# TEST FEATURE CONFIG
# =============================================================================


class TestFeatureConfig:
    """Test FeatureConfig dataclass."""

    def test_default_config(self):
        """Test default FeatureConfig values."""
        from services.futures_feature_flags import FeatureConfig, RolloutStage

        config = FeatureConfig()

        assert config.stage == RolloutStage.DISABLED
        assert config.canary_percentage == 0.0
        assert config.allowed_symbols is None
        assert config.allowed_accounts is None
        assert config.metadata == {}

    def test_custom_config(self):
        """Test custom FeatureConfig values."""
        from services.futures_feature_flags import FeatureConfig, RolloutStage

        config = FeatureConfig(
            stage=RolloutStage.CANARY,
            canary_percentage=25.0,
            allowed_symbols=["BTCUSDT", "ETHUSDT"],
            allowed_accounts=["account1"],
            metadata={"key": "value"},
        )

        assert config.stage == RolloutStage.CANARY
        assert config.canary_percentage == 25.0
        assert config.allowed_symbols == ["BTCUSDT", "ETHUSDT"]
        assert config.allowed_accounts == ["account1"]
        assert config.metadata == {"key": "value"}

    def test_canary_percentage_validation_valid(self):
        """Test valid canary percentage values."""
        from services.futures_feature_flags import FeatureConfig

        config_0 = FeatureConfig(canary_percentage=0.0)
        assert config_0.canary_percentage == 0.0

        config_50 = FeatureConfig(canary_percentage=50.0)
        assert config_50.canary_percentage == 50.0

        config_100 = FeatureConfig(canary_percentage=100.0)
        assert config_100.canary_percentage == 100.0

    def test_canary_percentage_validation_invalid(self):
        """Test invalid canary percentage values."""
        from services.futures_feature_flags import FeatureConfig

        with pytest.raises(ValueError):
            FeatureConfig(canary_percentage=-1.0)

        with pytest.raises(ValueError):
            FeatureConfig(canary_percentage=101.0)

        with pytest.raises(ValueError):
            FeatureConfig(canary_percentage=200.0)

    def test_stage_string_conversion(self):
        """Test that stage is converted from string."""
        from services.futures_feature_flags import FeatureConfig, RolloutStage

        # Stage can be initialized with string and converts to enum
        config = FeatureConfig(stage="production")
        assert config.stage == RolloutStage.PRODUCTION


# =============================================================================
# TEST FUTURES FEATURE FLAGS
# =============================================================================


class TestFuturesFeatureFlags:
    """Test FuturesFeatureFlags class."""

    def test_default_initialization(self):
        """Test default FuturesFeatureFlags initialization."""
        from services.futures_feature_flags import FuturesFeatureFlags

        flags = FuturesFeatureFlags()

        assert flags.global_kill_switch is False
        assert flags.environment == "development"

    def test_kill_switch_initialization(self):
        """Test FuturesFeatureFlags with kill switch enabled."""
        from services.futures_feature_flags import FuturesFeatureFlags

        flags = FuturesFeatureFlags(global_kill_switch=True)

        assert flags.global_kill_switch is True

    def test_set_stage_and_get_config(self):
        """Test setting stage and getting config."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags()

        # Set stage for a feature
        flags.set_stage(FuturesFeature.PERPETUAL_TRADING, RolloutStage.PRODUCTION)

        config = flags.get_config(FuturesFeature.PERPETUAL_TRADING)

        assert config.stage == RolloutStage.PRODUCTION

    def test_set_canary_percentage(self):
        """Test setting canary percentage."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.L3_EXECUTION, RolloutStage.CANARY)
        flags.set_canary_percentage(FuturesFeature.L3_EXECUTION, 50.0)

        config = flags.get_config(FuturesFeature.L3_EXECUTION)

        assert config.canary_percentage == 50.0

    def test_set_canary_percentage_invalid(self):
        """Test setting invalid canary percentage."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
        )

        flags = FuturesFeatureFlags()

        with pytest.raises(ValueError):
            flags.set_canary_percentage(FuturesFeature.L3_EXECUTION, 150.0)

    def test_get_nonexistent_feature_config(self):
        """Test getting config for feature that hasn't been explicitly set."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags()

        config = flags.get_config(FuturesFeature.COMMODITY_FUTURES)

        # Should return default (disabled) config
        assert config.stage == RolloutStage.DISABLED

    def test_get_stage(self):
        """Test get_stage method."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.PERPETUAL_TRADING, RolloutStage.PRODUCTION)

        stage = flags.get_stage(FuturesFeature.PERPETUAL_TRADING)
        assert stage == RolloutStage.PRODUCTION


# =============================================================================
# TEST IS_ENABLED METHOD
# =============================================================================


class TestIsEnabled:
    """Test is_enabled method."""

    def test_production_enabled(self):
        """Test that production stage is enabled."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.PERPETUAL_TRADING, RolloutStage.PRODUCTION)

        assert flags.is_enabled(FuturesFeature.PERPETUAL_TRADING) is True

    def test_disabled_not_enabled(self):
        """Test that disabled stage is not enabled."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.COMMODITY_FUTURES, RolloutStage.DISABLED)

        assert flags.is_enabled(FuturesFeature.COMMODITY_FUTURES) is False

    def test_shadow_is_enabled_for_comparison(self):
        """Test that shadow stage is enabled (for comparison/monitoring)."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.INDEX_FUTURES, RolloutStage.SHADOW)

        # Shadow should be enabled for running parallel comparison
        assert flags.is_enabled(FuturesFeature.INDEX_FUTURES) is True

    def test_canary_is_enabled(self):
        """Test that canary stage is enabled."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.L3_EXECUTION, RolloutStage.CANARY)

        # Canary should be enabled
        assert flags.is_enabled(FuturesFeature.L3_EXECUTION) is True

    def test_kill_switch_disables_all(self):
        """Test that global kill switch disables all features."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags(global_kill_switch=True)
        flags.set_stage(FuturesFeature.PERPETUAL_TRADING, RolloutStage.PRODUCTION)

        assert flags.is_enabled(FuturesFeature.PERPETUAL_TRADING) is False

    def test_is_production_method(self):
        """Test is_production method."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.PERPETUAL_TRADING, RolloutStage.PRODUCTION)
        flags.set_stage(FuturesFeature.L3_EXECUTION, RolloutStage.CANARY)

        assert flags.is_production(FuturesFeature.PERPETUAL_TRADING) is True
        assert flags.is_production(FuturesFeature.L3_EXECUTION) is False

    def test_is_shadow_mode_method(self):
        """Test is_shadow_mode method."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.INDEX_FUTURES, RolloutStage.SHADOW)

        assert flags.is_shadow_mode(FuturesFeature.INDEX_FUTURES) is True
        assert flags.is_shadow_mode(FuturesFeature.PERPETUAL_TRADING) is False

    def test_is_canary_method(self):
        """Test is_canary method."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.L3_EXECUTION, RolloutStage.CANARY)

        assert flags.is_canary(FuturesFeature.L3_EXECUTION) is True
        assert flags.is_canary(FuturesFeature.PERPETUAL_TRADING) is False


# =============================================================================
# TEST SHOULD_EXECUTE METHOD
# =============================================================================


class TestShouldExecute:
    """Test should_execute method with canary and filtering."""

    def test_production_always_executes(self):
        """Test that production stage always executes."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.PERPETUAL_TRADING, RolloutStage.PRODUCTION)

        # Should always execute for production
        for _ in range(100):
            assert flags.should_execute(FuturesFeature.PERPETUAL_TRADING) is True

    def test_disabled_never_executes(self):
        """Test that disabled stage never executes."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.COMMODITY_FUTURES, RolloutStage.DISABLED)

        # Should never execute for disabled
        for _ in range(100):
            assert flags.should_execute(FuturesFeature.COMMODITY_FUTURES) is False

    def test_canary_percentage_with_random_value(self):
        """Test canary with explicit random value."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.L3_EXECUTION, RolloutStage.CANARY)
        flags.set_canary_percentage(FuturesFeature.L3_EXECUTION, 50.0)

        # random_value < canary_percentage should execute
        assert flags.should_execute(FuturesFeature.L3_EXECUTION, random_value=25.0) is True
        assert flags.should_execute(FuturesFeature.L3_EXECUTION, random_value=49.9) is True
        assert flags.should_execute(FuturesFeature.L3_EXECUTION, random_value=50.0) is False
        assert flags.should_execute(FuturesFeature.L3_EXECUTION, random_value=75.0) is False

    def test_symbol_filter_allowed(self):
        """Test that allowed symbols are executed."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.L3_EXECUTION, RolloutStage.CANARY)
        flags.set_canary_percentage(FuturesFeature.L3_EXECUTION, 100.0)
        # Modify allowed_symbols directly
        flags.features[FuturesFeature.L3_EXECUTION].allowed_symbols = ["BTCUSDT", "ETHUSDT"]

        assert flags.should_execute(FuturesFeature.L3_EXECUTION, symbol="BTCUSDT") is True
        assert flags.should_execute(FuturesFeature.L3_EXECUTION, symbol="ETHUSDT") is True

    def test_symbol_filter_blocked(self):
        """Test that non-allowed symbols are blocked."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.L3_EXECUTION, RolloutStage.CANARY)
        flags.set_canary_percentage(FuturesFeature.L3_EXECUTION, 100.0)
        flags.features[FuturesFeature.L3_EXECUTION].allowed_symbols = ["BTCUSDT", "ETHUSDT"]

        assert flags.should_execute(FuturesFeature.L3_EXECUTION, symbol="SOLUSDT") is False
        assert flags.should_execute(FuturesFeature.L3_EXECUTION, symbol="DOGEUSDT") is False

    def test_account_filter_allowed(self):
        """Test that allowed accounts are executed."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.LEVERAGE_ACTION_SPACE, RolloutStage.CANARY)
        flags.set_canary_percentage(FuturesFeature.LEVERAGE_ACTION_SPACE, 100.0)
        flags.features[FuturesFeature.LEVERAGE_ACTION_SPACE].allowed_accounts = [
            "account1",
            "account2",
        ]

        assert (
            flags.should_execute(FuturesFeature.LEVERAGE_ACTION_SPACE, account_id="account1")
            is True
        )

    def test_account_filter_blocked(self):
        """Test that non-allowed accounts are blocked."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.LEVERAGE_ACTION_SPACE, RolloutStage.CANARY)
        flags.set_canary_percentage(FuturesFeature.LEVERAGE_ACTION_SPACE, 100.0)
        flags.features[FuturesFeature.LEVERAGE_ACTION_SPACE].allowed_accounts = ["account1"]

        assert (
            flags.should_execute(FuturesFeature.LEVERAGE_ACTION_SPACE, account_id="account999")
            is False
        )

    def test_shadow_mode_executes(self):
        """Test that shadow mode executes."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.INDEX_FUTURES, RolloutStage.SHADOW)

        assert flags.should_execute(FuturesFeature.INDEX_FUTURES) is True


# =============================================================================
# TEST YAML LOADING
# =============================================================================


class TestYAMLLoading:
    """Test YAML configuration loading."""

    def test_load_feature_flags(self, flags_yaml_path):
        """Test loading feature flags from YAML."""
        from services.futures_feature_flags import FuturesFeatureFlags

        flags = FuturesFeatureFlags.load(str(flags_yaml_path))

        assert flags is not None
        assert flags.global_kill_switch is False
        assert flags.environment == "staging"

    def test_loaded_perpetual_trading(self, flags_yaml_path):
        """Test perpetual trading config from YAML."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags.load(str(flags_yaml_path))
        config = flags.get_config(FuturesFeature.PERPETUAL_TRADING)

        assert config.stage == RolloutStage.PRODUCTION
        assert config.canary_percentage == 100

    def test_loaded_canary_feature(self, flags_yaml_path):
        """Test canary feature config from YAML."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags.load(str(flags_yaml_path))
        config = flags.get_config(FuturesFeature.QUARTERLY_TRADING)

        assert config.stage == RolloutStage.CANARY
        assert config.canary_percentage == 25
        assert config.allowed_symbols == ["BTCUSDT_QUARTERLY", "ETHUSDT_QUARTERLY"]

    def test_loaded_shadow_feature(self, flags_yaml_path):
        """Test shadow feature config from YAML."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags.load(str(flags_yaml_path))
        config = flags.get_config(FuturesFeature.INDEX_FUTURES)

        assert config.stage == RolloutStage.SHADOW
        assert config.allowed_symbols == ["ES", "NQ"]

    def test_loaded_disabled_feature(self, flags_yaml_path):
        """Test disabled feature config from YAML."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags.load(str(flags_yaml_path))
        config = flags.get_config(FuturesFeature.COMMODITY_FUTURES)

        assert config.stage == RolloutStage.DISABLED

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file returns defaults."""
        from services.futures_feature_flags import FuturesFeatureFlags

        # Should return default flags (not raise)
        flags = FuturesFeatureFlags.load("/nonexistent/path.yaml")
        assert flags is not None
        assert flags.global_kill_switch is False


# =============================================================================
# TEST GLOBAL FLAGS
# =============================================================================


class TestGlobalFlags:
    """Test global flags singleton."""

    def test_set_and_get_global_flags(self):
        """Test setting and getting global flags."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            get_global_flags,
            set_global_flags,
        )

        new_flags = FuturesFeatureFlags(global_kill_switch=True)
        set_global_flags(new_flags)

        retrieved = get_global_flags()
        assert retrieved.global_kill_switch is True

    def test_get_global_flags_creates_default(self):
        """Test that get_global_flags creates default if not set."""
        from services.futures_feature_flags import (
            get_global_flags,
            reset_global_flags,
        )

        reset_global_flags()
        flags = get_global_flags()

        assert flags is not None
        assert flags.global_kill_switch is False

    def test_init_feature_flags(self, flags_yaml_path):
        """Test init_feature_flags function."""
        from services.futures_feature_flags import (
            init_feature_flags,
            get_global_flags,
        )

        flags = init_feature_flags(str(flags_yaml_path))

        assert flags is not None
        assert flags.environment == "staging"

        # Should also be set as global
        global_flags = get_global_flags()
        assert global_flags.environment == "staging"


# =============================================================================
# TEST FEATURE FLAG DECORATOR
# =============================================================================


class TestFeatureFlagDecorator:
    """Test feature_flag decorator."""

    def test_decorator_with_enabled_feature(self):
        """Test decorator with enabled feature."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
            set_global_flags,
            feature_flag,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.PERPETUAL_TRADING, RolloutStage.PRODUCTION)
        set_global_flags(flags)

        @feature_flag(FuturesFeature.PERPETUAL_TRADING)
        def my_function():
            return "executed"

        result = my_function()
        assert result == "executed"

    def test_decorator_with_disabled_feature(self):
        """Test decorator with disabled feature."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
            set_global_flags,
            feature_flag,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.COMMODITY_FUTURES, RolloutStage.DISABLED)
        set_global_flags(flags)

        @feature_flag(FuturesFeature.COMMODITY_FUTURES)
        def my_function():
            return "executed"

        result = my_function()
        assert result is None  # Should return None when disabled

    def test_decorator_with_fallback(self):
        """Test decorator with fallback function."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
            set_global_flags,
            feature_flag,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.COMMODITY_FUTURES, RolloutStage.DISABLED)
        set_global_flags(flags)

        def fallback_function():
            return "fallback"

        @feature_flag(FuturesFeature.COMMODITY_FUTURES, fallback=fallback_function)
        def my_function():
            return "executed"

        result = my_function()
        assert result == "fallback"


# =============================================================================
# TEST REQUIRE FEATURE DECORATOR
# =============================================================================


class TestRequireFeatureDecorator:
    """Test require_feature decorator."""

    def test_require_enabled_feature(self):
        """Test require_feature with enabled feature."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
            set_global_flags,
            require_feature,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.PERPETUAL_TRADING, RolloutStage.PRODUCTION)
        set_global_flags(flags)

        @require_feature(FuturesFeature.PERPETUAL_TRADING)
        def my_function():
            return "executed"

        result = my_function()
        assert result == "executed"

    def test_require_disabled_feature_raises(self):
        """Test require_feature with disabled feature raises error."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
            set_global_flags,
            require_feature,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.COMMODITY_FUTURES, RolloutStage.DISABLED)
        set_global_flags(flags)

        @require_feature(FuturesFeature.COMMODITY_FUTURES)
        def my_function():
            return "executed"

        with pytest.raises(RuntimeError) as exc_info:
            my_function()

        assert "commodity_futures" in str(exc_info.value).lower()


# =============================================================================
# TEST THREAD SAFETY
# =============================================================================


class TestThreadSafety:
    """Test thread safety of feature flags."""

    def test_concurrent_access(self):
        """Test concurrent access to feature flags."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
            set_global_flags,
            get_global_flags,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.PERPETUAL_TRADING, RolloutStage.PRODUCTION)
        set_global_flags(flags)

        results = []
        errors = []

        def check_flag():
            try:
                for _ in range(100):
                    g_flags = get_global_flags()
                    result = g_flags.is_enabled(FuturesFeature.PERPETUAL_TRADING)
                    results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=check_flag) for _ in range(10)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(r is True for r in results)

    def test_concurrent_modification(self):
        """Test concurrent modification of feature flags."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags()
        errors = []

        def toggle_feature():
            try:
                for _ in range(50):
                    flags.set_stage(FuturesFeature.L3_EXECUTION, RolloutStage.PRODUCTION)
                    _ = flags.is_enabled(FuturesFeature.L3_EXECUTION)
                    flags.set_stage(FuturesFeature.L3_EXECUTION, RolloutStage.DISABLED)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=toggle_feature) for _ in range(5)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# TEST EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_none_symbol_with_filter(self):
        """Test with None symbol when filter is set."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.L3_EXECUTION, RolloutStage.CANARY)
        flags.set_canary_percentage(FuturesFeature.L3_EXECUTION, 100.0)
        flags.features[FuturesFeature.L3_EXECUTION].allowed_symbols = ["BTCUSDT"]

        # None symbol should still work (treated as not matching filter)
        result = flags.should_execute(FuturesFeature.L3_EXECUTION, symbol=None)
        # When symbol is None and allowed_symbols is set, it should check:
        # "symbol in allowed_symbols" -> "None in ['BTCUSDT']" -> False
        # Actually looking at the code: `if config.allowed_symbols and symbol and symbol not in config.allowed_symbols`
        # So if symbol is None, the check is skipped
        assert result is True  # Symbol=None skips the symbol check

    def test_canary_0_percent(self):
        """Test canary with 0% percentage."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.L3_EXECUTION, RolloutStage.CANARY)
        flags.set_canary_percentage(FuturesFeature.L3_EXECUTION, 0.0)

        # 0% canary should never execute when random_value provided
        for rv in [0, 25, 50, 75, 100]:
            assert (
                flags.should_execute(FuturesFeature.L3_EXECUTION, random_value=float(rv)) is False
            )

    def test_canary_100_percent(self):
        """Test canary with 100% percentage."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.L3_EXECUTION, RolloutStage.CANARY)
        flags.set_canary_percentage(FuturesFeature.L3_EXECUTION, 100.0)

        # 100% canary should always execute when random_value < 100
        for rv in [0, 25, 50, 75, 99.9]:
            assert flags.should_execute(FuturesFeature.L3_EXECUTION, random_value=rv) is True

    def test_kill_switch_enable_disable(self):
        """Test enable and disable kill switch."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.PERPETUAL_TRADING, RolloutStage.PRODUCTION)

        assert flags.is_enabled(FuturesFeature.PERPETUAL_TRADING) is True

        flags.enable_kill_switch()
        assert flags.is_enabled(FuturesFeature.PERPETUAL_TRADING) is False

        flags.disable_kill_switch()
        assert flags.is_enabled(FuturesFeature.PERPETUAL_TRADING) is True


# =============================================================================
# TEST HELPER FUNCTIONS
# =============================================================================


class TestHelperFunctions:
    """Test helper functions."""

    def test_enable_all_for_testing(self):
        """Test enable_all_for_testing helper."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
            enable_all_for_testing,
        )

        flags = enable_all_for_testing()

        # All features should be in production
        for feature in FuturesFeature:
            assert flags.get_stage(feature) == RolloutStage.PRODUCTION

    def test_create_minimal_crypto_flags(self):
        """Test create_minimal_crypto_flags helper."""
        from services.futures_feature_flags import (
            FuturesFeature,
            RolloutStage,
            create_minimal_crypto_flags,
        )

        flags = create_minimal_crypto_flags()

        # Essential crypto features should be enabled
        assert flags.get_stage(FuturesFeature.PERPETUAL_TRADING) == RolloutStage.PRODUCTION
        assert flags.get_stage(FuturesFeature.CROSS_MARGIN) == RolloutStage.PRODUCTION
        assert flags.get_stage(FuturesFeature.FUTURES_ENV_WRAPPER) == RolloutStage.PRODUCTION

    def test_create_minimal_cme_flags(self):
        """Test create_minimal_cme_flags helper."""
        from services.futures_feature_flags import (
            FuturesFeature,
            RolloutStage,
            create_minimal_cme_flags,
        )

        flags = create_minimal_cme_flags()

        # Essential CME features should be enabled
        assert flags.get_stage(FuturesFeature.INDEX_FUTURES) == RolloutStage.PRODUCTION
        assert flags.get_stage(FuturesFeature.SPAN_MARGIN) == RolloutStage.PRODUCTION
        assert flags.get_stage(FuturesFeature.DAILY_SETTLEMENT) == RolloutStage.PRODUCTION

    def test_get_enabled_features(self):
        """Test get_enabled_features method."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.PERPETUAL_TRADING, RolloutStage.PRODUCTION)
        flags.set_stage(FuturesFeature.L3_EXECUTION, RolloutStage.CANARY)

        enabled = flags.get_enabled_features()

        assert FuturesFeature.PERPETUAL_TRADING in enabled
        assert FuturesFeature.L3_EXECUTION in enabled
        assert FuturesFeature.COMMODITY_FUTURES not in enabled

    def test_get_production_features(self):
        """Test get_production_features method."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags()
        flags.set_stage(FuturesFeature.PERPETUAL_TRADING, RolloutStage.PRODUCTION)
        flags.set_stage(FuturesFeature.L3_EXECUTION, RolloutStage.CANARY)

        production = flags.get_production_features()

        assert FuturesFeature.PERPETUAL_TRADING in production
        assert FuturesFeature.L3_EXECUTION not in production


# =============================================================================
# TEST SERIALIZATION
# =============================================================================


class TestSerialization:
    """Test serialization and deserialization."""

    def test_to_dict(self):
        """Test to_dict method."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags(environment="production")
        flags.set_stage(FuturesFeature.PERPETUAL_TRADING, RolloutStage.PRODUCTION)

        data = flags.to_dict()

        assert data["environment"] == "production"
        assert data["global_kill_switch"] is False
        assert "features" in data
        assert data["features"]["perpetual_trading"]["stage"] == "production"

    def test_save_and_load(self, tmp_path):
        """Test save and load cycle."""
        from services.futures_feature_flags import (
            FuturesFeatureFlags,
            FuturesFeature,
            RolloutStage,
        )

        flags = FuturesFeatureFlags(environment="staging")
        flags.set_stage(FuturesFeature.PERPETUAL_TRADING, RolloutStage.PRODUCTION)
        flags.set_stage(FuturesFeature.L3_EXECUTION, RolloutStage.CANARY)
        flags.set_canary_percentage(FuturesFeature.L3_EXECUTION, 50.0)

        # Save
        save_path = tmp_path / "flags.yaml"
        flags.save(str(save_path))

        # Load
        loaded = FuturesFeatureFlags.load(str(save_path))

        assert loaded.environment == "staging"
        assert loaded.get_stage(FuturesFeature.PERPETUAL_TRADING) == RolloutStage.PRODUCTION
        assert loaded.get_stage(FuturesFeature.L3_EXECUTION) == RolloutStage.CANARY
        assert loaded.get_config(FuturesFeature.L3_EXECUTION).canary_percentage == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
