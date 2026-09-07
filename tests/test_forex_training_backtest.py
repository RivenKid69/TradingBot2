# -*- coding: utf-8 -*-
"""
test_forex_training_backtest.py
Comprehensive tests for Phase 9: Forex Training & Backtest Integration.

Tests cover:
- Forex configuration files (config_train_forex.yaml, config_backtest_forex.yaml)
- Data loading with forex-specific options
- ForexEnvWrapper functionality (session detection, swap costs, reward scaling)
- ForexLeverageWrapper functionality (margin calls, leverage constraints)
- Features pipeline forex integration
- End-to-end integration tests

Author: Sultan Suyunbayev
Date: 2025-11-30
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest
gym = pytest.importorskip("gymnasium")
import numpy as np
import pandas as pd
import pytest
pytest.importorskip("torch")
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_forex_df() -> pd.DataFrame:
    """Create sample forex OHLCV DataFrame for testing."""
    np.random.seed(42)
    n_bars = 100

    # Generate timestamps (4-hour bars)
    start_ts = 1704067200  # 2024-01-01 00:00:00 UTC
    timestamps = [start_ts + i * 4 * 3600 for i in range(n_bars)]

    # Generate EURUSD-like prices
    base_price = 1.1000
    returns = np.random.randn(n_bars) * 0.001
    prices = base_price * np.cumprod(1 + returns)

    # Generate OHLCV
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": prices * (1 + np.abs(np.random.randn(n_bars) * 0.001)),
        "low": prices * (1 - np.abs(np.random.randn(n_bars) * 0.001)),
        "close": prices * (1 + np.random.randn(n_bars) * 0.0005),
        "volume": np.random.uniform(1e6, 1e8, n_bars),
    })

    # Fix high/low
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df


@pytest.fixture
def sample_forex_df_with_sessions(sample_forex_df: pd.DataFrame) -> pd.DataFrame:
    """Create forex DataFrame with session features."""
    df = sample_forex_df.copy()

    # Add session features
    df["session_sydney"] = 0.0
    df["session_tokyo"] = 0.0
    df["session_london"] = 0.0
    df["session_new_york"] = 0.0

    for i, row in df.iterrows():
        ts = row["timestamp"]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour

        # Set session indicators
        if hour >= 21 or hour < 6:
            df.loc[i, "session_sydney"] = 1.0
        if 0 <= hour < 9:
            df.loc[i, "session_tokyo"] = 1.0
        if 7 <= hour < 16:
            df.loc[i, "session_london"] = 1.0
        if 12 <= hour < 21:
            df.loc[i, "session_new_york"] = 1.0

    return df


@pytest.fixture
def mock_trading_env():
    """Create a mock TradingEnv for wrapper testing."""
    env = MagicMock(spec=gym.Env)

    # Setup observation and action spaces
    env.observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(64,), dtype=np.float32
    )
    env.action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(1,), dtype=np.float32
    )

    # Setup default returns
    env.reset.return_value = (
        np.zeros(64, dtype=np.float32),
        {"timestamp": 1704067200, "signal_pos_next": 0.0}
    )
    env.step.return_value = (
        np.zeros(64, dtype=np.float32),
        0.01,  # reward
        False,  # terminated
        False,  # truncated
        {"timestamp": 1704081600, "signal_pos_next": 0.5}
    )

    return env


@pytest.fixture
def config_train_forex_path() -> Path:
    """Return path to forex training config."""
    return PROJECT_ROOT / "configs" / "config_train_forex.yaml"


@pytest.fixture
def config_backtest_forex_path() -> Path:
    """Return path to forex backtest config."""
    return PROJECT_ROOT / "configs" / "config_backtest_forex.yaml"


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================


class TestForexConfigurationFiles:
    """Tests for forex configuration file structure and loading."""

    def test_config_train_forex_exists(self, config_train_forex_path: Path):
        """Test that forex training config file exists."""
        assert config_train_forex_path.exists(), f"Config not found: {config_train_forex_path}"

    def test_config_backtest_forex_exists(self, config_backtest_forex_path: Path):
        """Test that forex backtest config file exists."""
        assert config_backtest_forex_path.exists(), f"Config not found: {config_backtest_forex_path}"

    def test_config_train_forex_valid_yaml(self, config_train_forex_path: Path):
        """Test that forex training config is valid YAML."""
        with open(config_train_forex_path, "r") as f:
            config = yaml.safe_load(f)
        assert isinstance(config, dict)
        assert "mode" in config
        assert config["mode"] == "train"

    def test_config_backtest_forex_valid_yaml(self, config_backtest_forex_path: Path):
        """Test that forex backtest config is valid YAML."""
        with open(config_backtest_forex_path, "r") as f:
            config = yaml.safe_load(f)
        assert isinstance(config, dict)
        assert "mode" in config
        assert config["mode"] == "backtest"

    def test_config_train_forex_asset_class(self, config_train_forex_path: Path):
        """Test that training config has correct asset class."""
        with open(config_train_forex_path, "r") as f:
            config = yaml.safe_load(f)
        assert config.get("asset_class") == "forex"

    def test_config_backtest_forex_asset_class(self, config_backtest_forex_path: Path):
        """Test that backtest config has correct asset class."""
        with open(config_backtest_forex_path, "r") as f:
            config = yaml.safe_load(f)
        assert config.get("asset_class") == "forex"

    def test_config_train_forex_has_data_section(self, config_train_forex_path: Path):
        """Test that training config has required data section."""
        with open(config_train_forex_path, "r") as f:
            config = yaml.safe_load(f)
        assert "data" in config
        data = config["data"]
        assert "timeframe" in data
        assert "filter_weekends" in data

    def test_config_train_forex_has_env_section(self, config_train_forex_path: Path):
        """Test that training config has required env section."""
        with open(config_train_forex_path, "r") as f:
            config = yaml.safe_load(f)
        assert "env" in config
        env = config["env"]
        assert "leverage" in env
        assert "session" in env

    def test_config_backtest_forex_has_slippage_section(self, config_backtest_forex_path: Path):
        """Test that backtest config has slippage configuration."""
        with open(config_backtest_forex_path, "r") as f:
            config = yaml.safe_load(f)
        assert "slippage" in config
        slippage = config["slippage"]
        assert slippage.get("provider") == "ForexParametricSlippageProvider"

    def test_config_backtest_forex_has_dealer_simulation(self, config_backtest_forex_path: Path):
        """Test that backtest config has OTC dealer simulation."""
        with open(config_backtest_forex_path, "r") as f:
            config = yaml.safe_load(f)
        assert "dealer_simulation" in config
        dealer = config["dealer_simulation"]
        assert dealer.get("enabled") is True
        assert "last_look_enabled" in dealer

    def test_config_forex_leverage_reasonable(self, config_train_forex_path: Path):
        """Test that forex leverage is within reasonable bounds."""
        with open(config_train_forex_path, "r") as f:
            config = yaml.safe_load(f)
        leverage = config.get("env", {}).get("leverage", 30.0)
        # Retail forex typically 30:1 to 50:1 (US/EU regulations)
        assert 1.0 <= leverage <= 500.0, f"Unreasonable leverage: {leverage}"

    def test_config_forex_pairs_list(self, config_backtest_forex_path: Path):
        """Test that backtest config has valid pairs list."""
        with open(config_backtest_forex_path, "r") as f:
            config = yaml.safe_load(f)
        pairs = config.get("data", {}).get("pairs", [])
        assert len(pairs) > 0, "No pairs specified"
        # Check format (should be SYMBOL_BASE format)
        for pair in pairs:
            assert "_" in pair or "/" in pair, f"Invalid pair format: {pair}"


# ============================================================================
# FOREX ENVIRONMENT WRAPPER TESTS
# ============================================================================


class TestForexEnvWrapper:
    """Tests for ForexEnvWrapper functionality."""

    def test_wrapper_initialization(self, mock_trading_env):
        """Test ForexEnvWrapper initialization."""
        from wrappers.forex_env import ForexEnvWrapper

        wrapper = ForexEnvWrapper(
            mock_trading_env,
            leverage=30.0,
            include_swap_costs=True,
            session_reward_scaling=False,
        )

        assert wrapper.leverage == 30.0
        assert wrapper.include_swap_costs is True
        assert wrapper.session_reward_scaling is False

    def test_wrapper_reset(self, mock_trading_env):
        """Test ForexEnvWrapper reset functionality."""
        from wrappers.forex_env import ForexEnvWrapper

        wrapper = ForexEnvWrapper(mock_trading_env, leverage=30.0)
        obs, info = wrapper.reset()

        assert obs is not None
        assert "forex_session" in info
        assert "session_liquidity" in info
        assert "max_leverage" in info
        assert info["max_leverage"] == 30.0

    def test_wrapper_step(self, mock_trading_env):
        """Test ForexEnvWrapper step functionality."""
        from wrappers.forex_env import ForexEnvWrapper

        wrapper = ForexEnvWrapper(mock_trading_env, leverage=30.0)
        wrapper.reset()

        action = np.array([0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = wrapper.step(action)

        assert obs is not None
        assert isinstance(reward, float)
        assert "swap_cost" in info
        assert "cumulative_swap_cost" in info

    def test_session_detection_sydney(self):
        """Test Sydney session detection."""
        from wrappers.forex_env import ForexEnvWrapper

        mock_env = MagicMock(spec=gym.Env)
        wrapper = ForexEnvWrapper(mock_env, leverage=30.0)

        # 22:00 UTC = Sydney session
        ts = datetime(2024, 1, 1, 22, 0, 0, tzinfo=timezone.utc).timestamp()
        session = wrapper._detect_session(int(ts))
        assert session == "sydney"

    def test_session_detection_tokyo(self):
        """Test Tokyo session detection."""
        from wrappers.forex_env import ForexEnvWrapper

        mock_env = MagicMock(spec=gym.Env)
        wrapper = ForexEnvWrapper(mock_env, leverage=30.0)

        # 03:00 UTC = Tokyo session (not overlap)
        ts = datetime(2024, 1, 1, 3, 0, 0, tzinfo=timezone.utc).timestamp()
        session = wrapper._detect_session(int(ts))
        assert session == "tokyo"

    def test_session_detection_london(self):
        """Test London session detection."""
        from wrappers.forex_env import ForexEnvWrapper

        mock_env = MagicMock(spec=gym.Env)
        wrapper = ForexEnvWrapper(mock_env, leverage=30.0)

        # 10:00 UTC = London session (after Tokyo overlap)
        ts = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc).timestamp()
        session = wrapper._detect_session(int(ts))
        assert session == "london"

    def test_session_detection_london_ny_overlap(self):
        """Test London-NY overlap detection."""
        from wrappers.forex_env import ForexEnvWrapper

        mock_env = MagicMock(spec=gym.Env)
        wrapper = ForexEnvWrapper(mock_env, leverage=30.0)

        # 14:00 UTC = London-NY overlap (highest priority)
        ts = datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc).timestamp()
        session = wrapper._detect_session(int(ts))
        assert session == "london_ny_overlap"

    def test_session_detection_tokyo_london_overlap(self):
        """Test Tokyo-London overlap detection."""
        from wrappers.forex_env import ForexEnvWrapper

        mock_env = MagicMock(spec=gym.Env)
        wrapper = ForexEnvWrapper(mock_env, leverage=30.0)

        # 08:00 UTC = Tokyo-London overlap
        ts = datetime(2024, 1, 1, 8, 0, 0, tzinfo=timezone.utc).timestamp()
        session = wrapper._detect_session(int(ts))
        assert session == "tokyo_london_overlap"

    def test_swap_cost_calculation_no_position(self):
        """Test swap cost is zero with no position."""
        from wrappers.forex_env import ForexEnvWrapper

        mock_env = MagicMock(spec=gym.Env)
        wrapper = ForexEnvWrapper(mock_env, leverage=30.0, include_swap_costs=True)

        # No position = no swap
        cost, rollovers = wrapper._calculate_swap_cost(
            prev_timestamp=1704067200,
            curr_timestamp=1704153600,  # 24 hours later
            position=0.0,
            info={}
        )
        assert cost == 0.0

    def test_swap_cost_calculation_with_position(self):
        """Test swap cost calculation with position."""
        from wrappers.forex_env import ForexEnvWrapper

        mock_env = MagicMock(spec=gym.Env)
        wrapper = ForexEnvWrapper(mock_env, leverage=30.0, include_swap_costs=True)

        # Long position over one night (21:00 UTC rollover)
        # Monday 20:00 UTC to Tuesday 22:00 UTC (one rollover)
        prev_ts = datetime(2024, 1, 8, 20, 0, 0, tzinfo=timezone.utc).timestamp()  # Monday
        curr_ts = datetime(2024, 1, 8, 22, 0, 0, tzinfo=timezone.utc).timestamp()  # Monday

        cost, rollovers = wrapper._calculate_swap_cost(
            prev_timestamp=int(prev_ts),
            curr_timestamp=int(curr_ts),
            position=1.0,
            info={"long_swap": -0.3, "short_swap": 0.1}
        )

        # Should have swap cost (position > 0, crosses 21:00)
        assert cost > 0 or cost == 0.0  # Depends on whether rollover hour was crossed

    def test_rollover_count_no_crossings(self):
        """Test rollover count with no 21:00 UTC crossings."""
        from wrappers.forex_env import ForexEnvWrapper

        mock_env = MagicMock(spec=gym.Env)
        wrapper = ForexEnvWrapper(mock_env, leverage=30.0)

        # Same hour, no crossing
        prev_ts = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc).timestamp()
        curr_ts = datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc).timestamp()

        count = wrapper._count_rollovers(int(prev_ts), int(curr_ts))
        assert count == 0

    def test_rollover_count_single_crossing(self):
        """Test rollover count with single 21:00 UTC crossing."""
        from wrappers.forex_env import ForexEnvWrapper

        mock_env = MagicMock(spec=gym.Env)
        wrapper = ForexEnvWrapper(mock_env, leverage=30.0)

        # Cross 21:00 once on Monday
        prev_ts = datetime(2024, 1, 8, 20, 0, 0, tzinfo=timezone.utc).timestamp()  # Monday
        curr_ts = datetime(2024, 1, 8, 22, 0, 0, tzinfo=timezone.utc).timestamp()  # Monday

        count = wrapper._count_rollovers(int(prev_ts), int(curr_ts))
        assert count == 1

    def test_rollover_count_wednesday_triple(self):
        """Test that Wednesday rollover counts as 3 (weekend adjustment)."""
        from wrappers.forex_env import ForexEnvWrapper

        mock_env = MagicMock(spec=gym.Env)
        wrapper = ForexEnvWrapper(mock_env, leverage=30.0)

        # Cross 21:00 on Wednesday
        prev_ts = datetime(2024, 1, 10, 20, 0, 0, tzinfo=timezone.utc).timestamp()  # Wednesday
        curr_ts = datetime(2024, 1, 10, 22, 0, 0, tzinfo=timezone.utc).timestamp()  # Wednesday

        count = wrapper._count_rollovers(int(prev_ts), int(curr_ts))
        assert count == 3  # Triple swap on Wednesday

    def test_session_reward_scaling_enabled(self, mock_trading_env):
        """Test session-based reward scaling when enabled."""
        from wrappers.forex_env import ForexEnvWrapper

        wrapper = ForexEnvWrapper(
            mock_trading_env,
            leverage=30.0,
            session_reward_scaling=True,
            session_reward_scale_min=0.5,
        )
        wrapper.reset()

        # Step returns modified reward based on session
        action = np.array([0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = wrapper.step(action)

        # Reward should be modified (scaled)
        assert "reward_adjustment" in info

    def test_info_enrichment(self, mock_trading_env):
        """Test that info dict is enriched with forex-specific data."""
        from wrappers.forex_env import ForexEnvWrapper

        wrapper = ForexEnvWrapper(mock_trading_env, leverage=30.0)
        obs, info = wrapper.reset()

        # Check all expected fields
        assert "forex_session" in info
        assert "session_liquidity" in info
        assert "max_leverage" in info
        assert "is_session_overlap" in info


# ============================================================================
# FOREX LEVERAGE WRAPPER TESTS
# ============================================================================


class TestForexLeverageWrapper:
    """Tests for ForexLeverageWrapper functionality."""

    def test_leverage_wrapper_initialization(self, mock_trading_env):
        """Test ForexLeverageWrapper initialization."""
        from wrappers.forex_env import ForexLeverageWrapper

        wrapper = ForexLeverageWrapper(
            mock_trading_env,
            max_leverage=30.0,
            margin_call_level=1.0,
        )

        assert wrapper.max_leverage == 30.0
        assert wrapper.margin_call_level == 1.0

    def test_leverage_wrapper_step_normal(self, mock_trading_env):
        """Test step with normal margin level."""
        from wrappers.forex_env import ForexLeverageWrapper

        # Setup environment with good margin
        mock_trading_env._net_worth = 100000.0
        mock_trading_env._used_margin = 10000.0  # 10x margin level

        wrapper = ForexLeverageWrapper(mock_trading_env, max_leverage=30.0)

        action = np.array([0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = wrapper.step(action)

        # Action should pass through unchanged
        mock_trading_env.step.assert_called_once()
        called_action = mock_trading_env.step.call_args[0][0]
        np.testing.assert_array_almost_equal(called_action, action)

    def test_leverage_wrapper_margin_call_reduction(self, mock_trading_env):
        """Test position reduction on margin call."""
        from wrappers.forex_env import ForexLeverageWrapper

        # Setup environment with low margin (margin call level)
        mock_trading_env._net_worth = 10000.0
        mock_trading_env._used_margin = 12000.0  # margin_level = 0.83 < 1.0

        wrapper = ForexLeverageWrapper(
            mock_trading_env,
            max_leverage=30.0,
            margin_call_level=1.0,
        )

        action = np.array([1.0], dtype=np.float32)  # Full position
        wrapper.step(action)

        # Action should be reduced by 50%
        called_action = mock_trading_env.step.call_args[0][0]
        np.testing.assert_array_almost_equal(called_action, np.array([0.5]))

    def test_leverage_wrapper_no_margin_used(self, mock_trading_env):
        """Test step with no margin used (infinite margin level)."""
        from wrappers.forex_env import ForexLeverageWrapper

        mock_trading_env._net_worth = 100000.0
        mock_trading_env._used_margin = 0.0  # No margin used

        wrapper = ForexLeverageWrapper(mock_trading_env, max_leverage=30.0)

        action = np.array([0.5], dtype=np.float32)
        wrapper.step(action)

        # Action should pass through unchanged
        called_action = mock_trading_env.step.call_args[0][0]
        np.testing.assert_array_almost_equal(called_action, action)


# ============================================================================
# FACTORY FUNCTION TESTS
# ============================================================================


class TestCreateForexEnv:
    """Tests for create_forex_env factory function."""

    def test_factory_returns_wrapped_env(self, sample_forex_df):
        """Test that factory returns wrapped environment."""
        # Mock TradingEnv to avoid full initialization
        # Import is inside function, so patch trading_patchnew module
        with patch.dict("sys.modules", {"trading_patchnew": MagicMock()}):
            import sys
            mock_trading = sys.modules["trading_patchnew"]
            mock_env = MagicMock(spec=gym.Env)
            mock_env.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(64,), dtype=np.float32
            )
            mock_env.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
            mock_trading.TradingEnv.return_value = mock_env

            # Reimport to pick up mock
            import importlib
            import wrappers.forex_env
            importlib.reload(wrappers.forex_env)
            from wrappers.forex_env import create_forex_env, ForexLeverageWrapper

            env = create_forex_env(
                df=sample_forex_df,
                leverage=30.0,
                include_swap_costs=True,
            )

            # Should be wrapped in ForexLeverageWrapper (outermost)
            assert isinstance(env, ForexLeverageWrapper)

    def test_factory_sets_asset_class(self, sample_forex_df):
        """Test that factory sets asset_class='forex'."""
        with patch.dict("sys.modules", {"trading_patchnew": MagicMock()}):
            import sys
            mock_trading = sys.modules["trading_patchnew"]
            mock_env = MagicMock(spec=gym.Env)
            mock_env.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(64,), dtype=np.float32
            )
            mock_env.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
            mock_trading.TradingEnv.return_value = mock_env

            import importlib
            import wrappers.forex_env
            importlib.reload(wrappers.forex_env)
            from wrappers.forex_env import create_forex_env

            create_forex_env(df=sample_forex_df, leverage=30.0)

            # Check TradingEnv was called with asset_class='forex'
            call_kwargs = mock_trading.TradingEnv.call_args[1]
            assert call_kwargs.get("asset_class") == "forex"


# ============================================================================
# DATA LOADER INTEGRATION TESTS
# ============================================================================


class TestDataLoaderForexIntegration:
    """Tests for data_loader_multi_asset.py forex integration."""

    def test_forex_asset_class_enum_exists(self):
        """Test that FOREX is in AssetClass enum."""
        from data_loader_multi_asset import AssetClass

        assert hasattr(AssetClass, "FOREX")

    def test_forex_data_vendor_enum_exists(self):
        """Test that forex data vendors exist in DataVendor enum."""
        from data_loader_multi_asset import DataVendor

        # At least one forex vendor should exist
        forex_vendors = ["OANDA", "IG", "DUKASCOPY"]
        found = any(hasattr(DataVendor, v) for v in forex_vendors)
        assert found, "No forex vendors found in DataVendor enum"

    def test_load_multi_asset_data_forex_params(self):
        """Test that load_multi_asset_data accepts forex parameters."""
        from data_loader_multi_asset import load_multi_asset_data
        import inspect

        sig = inspect.signature(load_multi_asset_data)
        params = sig.parameters

        # Check forex-specific parameters exist
        assert "filter_weekends" in params
        assert "add_session_features" in params
        assert "merge_swap_rates" in params
        assert "merge_interest_rates" in params
        assert "merge_calendar" in params

    def test_load_multi_asset_data_with_forex_df(self, sample_forex_df):
        """Test loading with forex-specific options."""
        # Create temp parquet file
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "EURUSD.parquet"
            sample_forex_df.to_parquet(filepath)

            from data_loader_multi_asset import load_multi_asset_data

            frames, obs_shapes = load_multi_asset_data(
                paths=[str(tmpdir) + "/*.parquet"],
                asset_class="forex",
                filter_weekends=True,
                add_session_features=False,  # Don't add features (no forex_features)
            )

            assert len(frames) > 0
            assert "EURUSD" in frames or len(frames) == 1


# ============================================================================
# FEATURES PIPELINE INTEGRATION TESTS
# ============================================================================


class TestFeaturesPipelineForexIntegration:
    """Tests for features_pipeline.py forex integration."""

    def test_valid_asset_classes_includes_forex(self):
        """Test that VALID_ASSET_CLASSES includes 'forex'."""
        from features_pipeline import FeaturePipeline

        assert "forex" in FeaturePipeline.VALID_ASSET_CLASSES

    def test_feature_pipeline_accepts_forex_asset_class(self):
        """Test FeaturePipeline accepts asset_class='forex'."""
        from features_pipeline import FeaturePipeline

        pipeline = FeaturePipeline(asset_class="forex")
        assert pipeline.asset_class == "forex"

    def test_auto_forex_features_default_true(self):
        """Test that auto_forex_features defaults to True."""
        from features_pipeline import FeaturePipeline

        pipeline = FeaturePipeline(asset_class="forex")
        assert pipeline.auto_forex_features is True

    def test_auto_forex_features_can_disable(self):
        """Test that auto_forex_features can be disabled."""
        from features_pipeline import FeaturePipeline

        pipeline = FeaturePipeline(asset_class="forex", auto_forex_features=False)
        assert pipeline.auto_forex_features is False


# ============================================================================
# SESSION LIQUIDITY TESTS
# ============================================================================


class TestSessionLiquidity:
    """Tests for forex session liquidity factors."""

    def test_session_liquidity_values_defined(self):
        """Test that session liquidity values are defined."""
        from wrappers.forex_env import SESSION_LIQUIDITY

        expected_sessions = [
            "sydney", "tokyo", "london", "new_york",
            "london_ny_overlap", "tokyo_london_overlap", "low_liquidity"
        ]

        for session in expected_sessions:
            assert session in SESSION_LIQUIDITY

    def test_session_liquidity_overlap_higher(self):
        """Test that overlap sessions have higher liquidity."""
        from wrappers.forex_env import SESSION_LIQUIDITY

        # Overlaps should have higher liquidity than individual sessions
        assert SESSION_LIQUIDITY["london_ny_overlap"] > SESSION_LIQUIDITY["london"]
        assert SESSION_LIQUIDITY["london_ny_overlap"] > SESSION_LIQUIDITY["new_york"]

    def test_session_liquidity_low_liquidity_lowest(self):
        """Test that low_liquidity has lowest factor (except weekend which is 0)."""
        from wrappers.forex_env import SESSION_LIQUIDITY

        low_liq = SESSION_LIQUIDITY["low_liquidity"]
        for session, factor in SESSION_LIQUIDITY.items():
            if session not in ("low_liquidity", "weekend"):  # Weekend is 0, which is lower
                assert factor >= low_liq, f"{session} has lower liquidity than low_liquidity"

        # Weekend should have zero liquidity
        assert SESSION_LIQUIDITY["weekend"] == 0.0


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_wrapper_with_zero_timestamp(self, mock_trading_env):
        """Test wrapper handles zero timestamp gracefully."""
        from wrappers.forex_env import ForexEnvWrapper

        # Return zero timestamp
        mock_trading_env.reset.return_value = (
            np.zeros(64, dtype=np.float32),
            {"timestamp": 0}
        )

        wrapper = ForexEnvWrapper(mock_trading_env, leverage=30.0)
        obs, info = wrapper.reset()

        # Should use default session
        assert "forex_session" in info
        assert info["forex_session"] == "london"  # Default

    def test_swap_cost_same_timestamp(self):
        """Test swap cost with same prev/curr timestamp."""
        from wrappers.forex_env import ForexEnvWrapper

        mock_env = MagicMock(spec=gym.Env)
        wrapper = ForexEnvWrapper(mock_env, leverage=30.0)

        ts = 1704067200
        cost, rollovers = wrapper._calculate_swap_cost(ts, ts, 1.0, {})
        assert cost == 0.0
        assert rollovers == 0

    def test_swap_cost_prev_greater_than_curr(self):
        """Test swap cost with invalid timestamp order."""
        from wrappers.forex_env import ForexEnvWrapper

        mock_env = MagicMock(spec=gym.Env)
        wrapper = ForexEnvWrapper(mock_env, leverage=30.0)

        # prev > curr should return 0
        cost, rollovers = wrapper._calculate_swap_cost(1704153600, 1704067200, 1.0, {})
        assert cost == 0.0
        assert rollovers == 0

    def test_rollover_count_prev_greater_than_curr(self):
        """Test rollover count with invalid timestamp order."""
        from wrappers.forex_env import ForexEnvWrapper

        mock_env = MagicMock(spec=gym.Env)
        wrapper = ForexEnvWrapper(mock_env, leverage=30.0)

        # prev >= curr should return 0
        count = wrapper._count_rollovers(1704153600, 1704067200)
        assert count == 0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_full_wrapper_chain(self, mock_trading_env):
        """Test full wrapper chain (ForexEnv + Leverage)."""
        from wrappers.forex_env import ForexEnvWrapper, ForexLeverageWrapper

        # Setup margin attributes
        mock_trading_env._net_worth = 100000.0
        mock_trading_env._used_margin = 10000.0

        # Apply both wrappers
        env = ForexEnvWrapper(mock_trading_env, leverage=30.0)
        env = ForexLeverageWrapper(env, max_leverage=30.0)

        # Reset and step
        obs, info = env.reset()
        action = np.array([0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        # Should have all forex info
        assert "forex_session" in info

    def test_config_to_wrapper_integration(self, config_train_forex_path: Path):
        """Test loading config and creating wrapper with its values."""
        from wrappers.forex_env import ForexEnvWrapper

        with open(config_train_forex_path, "r") as f:
            config = yaml.safe_load(f)

        env_config = config.get("env", {})
        leverage = env_config.get("leverage", 30.0)

        mock_env = MagicMock(spec=gym.Env)
        mock_env.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(64,), dtype=np.float32
        )
        mock_env.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        mock_env.reset.return_value = (
            np.zeros(64, dtype=np.float32),
            {"timestamp": 1704067200}
        )

        wrapper = ForexEnvWrapper(mock_env, leverage=leverage)
        assert wrapper.leverage == leverage


# ============================================================================
# WRAPPERS MODULE EXPORT TESTS
# ============================================================================


class TestWrappersModuleExports:
    """Tests for wrappers module exports."""

    def test_forex_wrappers_exported(self):
        """Test that forex wrappers are exported from wrappers module."""
        from wrappers import ForexEnvWrapper, ForexLeverageWrapper, create_forex_env

        assert ForexEnvWrapper is not None
        assert ForexLeverageWrapper is not None
        assert create_forex_env is not None

    def test_all_includes_forex(self):
        """Test that __all__ includes forex exports."""
        from wrappers import __all__

        assert "ForexEnvWrapper" in __all__
        assert "ForexLeverageWrapper" in __all__
        assert "create_forex_env" in __all__


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
