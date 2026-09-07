# -*- coding: utf-8 -*-
"""
tests/test_signal_only_stocks_config.py

Tests for signal-only stocks configuration and universe loading.

Test coverage:
- Signal-only stocks config validation
- Equity-specific config values
- Universe file loading (crypto and stocks)
- Backward compatibility with crypto configs
- Config field consistency between crypto and equity
"""

import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml


# =============================================================================
# Paths
# =============================================================================

ROOT_DIR = Path(__file__).parent.parent
CONFIGS_DIR = ROOT_DIR / "configs"
UNIVERSE_DIR = ROOT_DIR / "data" / "universe"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def signal_only_stocks_config_path() -> Path:
    """Path to signal-only stocks config."""
    return CONFIGS_DIR / "config_train_signal_only_stocks.yaml"


@pytest.fixture
def signal_only_crypto_config_path() -> Path:
    """Path to signal-only crypto config."""
    return CONFIGS_DIR / "config_train_signal_only_crypto.yaml"


@pytest.fixture
def stocks_config_path() -> Path:
    """Path to regular stocks training config."""
    return CONFIGS_DIR / "config_train_stocks.yaml"


@pytest.fixture
def signal_only_stocks_config(signal_only_stocks_config_path: Path) -> Dict[str, Any]:
    """Load signal-only stocks config."""
    with open(signal_only_stocks_config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def signal_only_crypto_config(signal_only_crypto_config_path: Path) -> Dict[str, Any]:
    """Load signal-only crypto config."""
    with open(signal_only_crypto_config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def stocks_config(stocks_config_path: Path) -> Dict[str, Any]:
    """Load regular stocks training config."""
    with open(stocks_config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def crypto_symbols_path() -> Path:
    """Path to crypto symbols file."""
    return UNIVERSE_DIR / "symbols.json"


@pytest.fixture
def alpaca_symbols_path() -> Path:
    """Path to Alpaca symbols file."""
    return UNIVERSE_DIR / "alpaca_symbols.json"


# =============================================================================
# Test Config File Existence
# =============================================================================


class TestConfigFileExistence:
    """Tests for config file existence."""

    def test_signal_only_stocks_config_exists(self, signal_only_stocks_config_path: Path):
        """Test that signal-only stocks config file exists."""
        assert (
            signal_only_stocks_config_path.exists()
        ), f"Config file not found: {signal_only_stocks_config_path}"

    def test_signal_only_crypto_config_exists(self, signal_only_crypto_config_path: Path):
        """Test that signal-only crypto config file exists."""
        assert (
            signal_only_crypto_config_path.exists()
        ), f"Config file not found: {signal_only_crypto_config_path}"

    def test_stocks_config_exists(self, stocks_config_path: Path):
        """Test that regular stocks config file exists."""
        assert stocks_config_path.exists(), f"Config file not found: {stocks_config_path}"


# =============================================================================
# Test Signal-Only Stocks Config Values
# =============================================================================


class TestSignalOnlyStocksConfig:
    """Tests for signal-only stocks configuration values."""

    def test_asset_class_is_equity(self, signal_only_stocks_config: Dict[str, Any]):
        """Test asset_class is set to equity."""
        assert signal_only_stocks_config["asset_class"] == "equity"

    def test_data_vendor_is_alpaca(self, signal_only_stocks_config: Dict[str, Any]):
        """Test data_vendor is set to alpaca."""
        assert signal_only_stocks_config["data_vendor"] == "alpaca"

    def test_market_is_equity(self, signal_only_stocks_config: Dict[str, Any]):
        """Test market is set to equity."""
        assert signal_only_stocks_config["market"] == "equity"

    def test_mode_is_train(self, signal_only_stocks_config: Dict[str, Any]):
        """Test mode is set to train."""
        assert signal_only_stocks_config["mode"] == "train"

    def test_tif_is_day(self, signal_only_stocks_config: Dict[str, Any]):
        """Test TIF is DAY for equity (not GTC like crypto)."""
        exec_params = signal_only_stocks_config.get("execution_params", {})
        assert exec_params.get("tif") == "DAY"

    def test_risk_is_disabled(self, signal_only_stocks_config: Dict[str, Any]):
        """Test risk is disabled for signal-only mode."""
        risk = signal_only_stocks_config.get("risk", {})
        assert risk.get("enabled") is False

    def test_no_trade_is_disabled(self, signal_only_stocks_config: Dict[str, Any]):
        """Test no_trade is disabled for signal-only mode."""
        env = signal_only_stocks_config.get("env", {})
        no_trade = env.get("no_trade", {})
        assert no_trade.get("enabled") is False

    def test_trading_hours_filter_enabled(self, signal_only_stocks_config: Dict[str, Any]):
        """Test filter_trading_hours is enabled for stocks."""
        data = signal_only_stocks_config.get("data", {})
        assert data.get("filter_trading_hours") is True

    def test_extended_hours_disabled(self, signal_only_stocks_config: Dict[str, Any]):
        """Test extended hours is disabled."""
        data = signal_only_stocks_config.get("data", {})
        assert data.get("include_extended_hours") is False

    def test_session_calendar_is_us_equity(self, signal_only_stocks_config: Dict[str, Any]):
        """Test session calendar is us_equity."""
        env = signal_only_stocks_config.get("env", {})
        session = env.get("session", {})
        assert session.get("calendar") == "us_equity"

    def test_fees_are_commission_free(self, signal_only_stocks_config: Dict[str, Any]):
        """Test fees structure is commission-free."""
        fees = signal_only_stocks_config.get("fees", {})
        assert fees.get("maker_bps") == 0.0
        assert fees.get("taker_bps") == 0.0

    def test_regulatory_fees_configured(self, signal_only_stocks_config: Dict[str, Any]):
        """Test regulatory fees are configured."""
        fees = signal_only_stocks_config.get("fees", {})
        regulatory = fees.get("regulatory", {})
        assert regulatory.get("enabled") is True
        assert regulatory.get("sec_fee_per_million") == 27.80
        assert regulatory.get("taf_fee_per_share") == 0.000166

    def test_slippage_lower_than_crypto(self, signal_only_stocks_config: Dict[str, Any]):
        """Test slippage k is lower than crypto (more liquid markets)."""
        slippage = signal_only_stocks_config.get("slippage", {})
        k = slippage.get("k", 1.0)
        assert k <= 0.5  # Crypto typically uses k=0.5-1.0

    def test_spread_bps_tighter_than_crypto(self, signal_only_stocks_config: Dict[str, Any]):
        """Test spread is tighter than crypto."""
        slippage = signal_only_stocks_config.get("slippage", {})
        spread = slippage.get("default_spread_bps", 5.0)
        assert spread <= 2.0  # Crypto typically 2-5 bps

    def test_turnover_penalty_lower_than_crypto(self, signal_only_stocks_config: Dict[str, Any]):
        """Test turnover penalty is lower (commission-free trading)."""
        model = signal_only_stocks_config.get("model", {})
        params = model.get("params", {})
        turnover = params.get("turnover_penalty_coef", 0.001)
        assert turnover <= 0.0001  # Crypto uses ~0.0003

    def test_long_only_action_space(self, signal_only_stocks_config: Dict[str, Any]):
        """Test long_only is enabled for stocks."""
        algo = signal_only_stocks_config.get("algo", {})
        actions = algo.get("actions", {})
        assert actions.get("long_only") is True

    def test_twin_critics_enabled(self, signal_only_stocks_config: Dict[str, Any]):
        """Test twin critics are enabled."""
        model = signal_only_stocks_config.get("model", {})
        params = model.get("params", {})
        assert params.get("use_twin_critics") is True

    def test_vgs_enabled(self, signal_only_stocks_config: Dict[str, Any]):
        """Test VGS is enabled."""
        model = signal_only_stocks_config.get("model", {})
        vgs = model.get("vgs", {})
        assert vgs.get("enabled") is True

    def test_adaptive_upgd_optimizer(self, signal_only_stocks_config: Dict[str, Any]):
        """Test AdaptiveUPGD is the optimizer."""
        model = signal_only_stocks_config.get("model", {})
        assert model.get("optimizer_class") == "AdaptiveUPGD"


# =============================================================================
# Test Consistency Between Crypto and Stocks Signal-Only Configs
# =============================================================================


class TestConfigConsistency:
    """Tests for consistency between crypto and stocks signal-only configs."""

    def test_both_have_same_mode(
        self,
        signal_only_stocks_config: Dict[str, Any],
        signal_only_crypto_config: Dict[str, Any],
    ):
        """Test both configs have mode=train."""
        assert signal_only_stocks_config["mode"] == "train"
        assert signal_only_crypto_config["mode"] == "train"

    def test_both_have_twin_critics(
        self,
        signal_only_stocks_config: Dict[str, Any],
        signal_only_crypto_config: Dict[str, Any],
    ):
        """Test both configs have twin critics enabled."""
        stocks_params = signal_only_stocks_config.get("model", {}).get("params", {})
        crypto_params = signal_only_crypto_config.get("model", {}).get("params", {})
        assert stocks_params.get("use_twin_critics") is True
        assert crypto_params.get("use_twin_critics") is True

    def test_both_have_vgs(
        self,
        signal_only_stocks_config: Dict[str, Any],
        signal_only_crypto_config: Dict[str, Any],
    ):
        """Test both configs have VGS enabled."""
        stocks_vgs = signal_only_stocks_config.get("model", {}).get("vgs", {})
        crypto_vgs = signal_only_crypto_config.get("model", {}).get("vgs", {})
        assert stocks_vgs.get("enabled") is True
        assert crypto_vgs.get("enabled") is True

    def test_both_have_adaptive_upgd(
        self,
        signal_only_stocks_config: Dict[str, Any],
        signal_only_crypto_config: Dict[str, Any],
    ):
        """Test both configs use AdaptiveUPGD."""
        stocks_model = signal_only_stocks_config.get("model", {})
        crypto_model = signal_only_crypto_config.get("model", {})
        assert stocks_model.get("optimizer_class") == "AdaptiveUPGD"
        assert crypto_model.get("optimizer_class") == "AdaptiveUPGD"

    def test_both_have_same_gamma(
        self,
        signal_only_stocks_config: Dict[str, Any],
        signal_only_crypto_config: Dict[str, Any],
    ):
        """Test both configs have same gamma for reward shaping consistency."""
        stocks_gamma = (
            signal_only_stocks_config.get("model", {}).get("params", {}).get("gamma", 0.0)
        )
        crypto_gamma = (
            signal_only_crypto_config.get("model", {}).get("params", {}).get("gamma", 0.0)
        )
        assert stocks_gamma == 0.99
        assert crypto_gamma == 0.99

    def test_both_have_long_only(
        self,
        signal_only_stocks_config: Dict[str, Any],
        signal_only_crypto_config: Dict[str, Any],
    ):
        """Test both configs have long_only action space."""
        stocks_long_only = (
            signal_only_stocks_config.get("algo", {}).get("actions", {}).get("long_only", False)
        )
        crypto_long_only = (
            signal_only_crypto_config.get("algo", {}).get("actions", {}).get("long_only", False)
        )
        assert stocks_long_only is True
        assert crypto_long_only is True

    def test_risk_disabled_both(
        self,
        signal_only_stocks_config: Dict[str, Any],
        signal_only_crypto_config: Dict[str, Any],
    ):
        """Test risk is disabled in both signal-only configs."""
        stocks_risk = signal_only_stocks_config.get("risk", {}).get("enabled", True)
        crypto_risk = signal_only_crypto_config.get("risk", {}).get("enabled", True)
        assert stocks_risk is False
        assert crypto_risk is False


# =============================================================================
# Test Equity-Specific Config Values Differ From Crypto
# =============================================================================


class TestEquityVsCryptoDifferences:
    """Tests for differences between equity and crypto configs."""

    def test_asset_class_different(
        self,
        signal_only_stocks_config: Dict[str, Any],
        signal_only_crypto_config: Dict[str, Any],
    ):
        """Test asset_class differs between configs."""
        assert signal_only_stocks_config["asset_class"] == "equity"
        assert signal_only_crypto_config["asset_class"] == "crypto"

    def test_data_vendor_different(
        self,
        signal_only_stocks_config: Dict[str, Any],
        signal_only_crypto_config: Dict[str, Any],
    ):
        """Test data_vendor differs between configs."""
        assert signal_only_stocks_config["data_vendor"] == "alpaca"
        assert signal_only_crypto_config["data_vendor"] == "binance"

    def test_tif_different(
        self,
        signal_only_stocks_config: Dict[str, Any],
        signal_only_crypto_config: Dict[str, Any],
    ):
        """Test TIF differs (DAY for stocks, GTC for crypto)."""
        stocks_tif = signal_only_stocks_config.get("execution_params", {}).get("tif")
        crypto_tif = signal_only_crypto_config.get("execution_params", {}).get("tif")
        assert stocks_tif == "DAY"
        assert crypto_tif == "GTC"

    def test_session_calendar_different(
        self,
        signal_only_stocks_config: Dict[str, Any],
        signal_only_crypto_config: Dict[str, Any],
    ):
        """Test session calendar is different."""
        stocks_session = signal_only_stocks_config.get("env", {}).get("session", {})
        # Crypto may not have session.calendar
        assert stocks_session.get("calendar") == "us_equity"


# =============================================================================
# Test Universe Files
# =============================================================================


class TestUniverseFiles:
    """Tests for universe symbol files."""

    def test_crypto_symbols_file_exists(self, crypto_symbols_path: Path):
        """Test crypto symbols file exists."""
        assert crypto_symbols_path.exists(), f"Crypto symbols file not found: {crypto_symbols_path}"

    def test_alpaca_symbols_file_exists(self, alpaca_symbols_path: Path):
        """Test Alpaca symbols file exists."""
        assert alpaca_symbols_path.exists(), f"Alpaca symbols file not found: {alpaca_symbols_path}"

    def test_crypto_symbols_is_valid_json(self, crypto_symbols_path: Path):
        """Test crypto symbols file is valid JSON."""
        with open(crypto_symbols_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)

    def test_alpaca_symbols_is_valid_json(self, alpaca_symbols_path: Path):
        """Test Alpaca symbols file is valid JSON."""
        with open(alpaca_symbols_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_crypto_symbols_not_empty(self, crypto_symbols_path: Path):
        """Test crypto symbols list is not empty."""
        with open(crypto_symbols_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) > 0, "Crypto symbols list is empty"

    def test_alpaca_symbols_not_empty(self, alpaca_symbols_path: Path):
        """Test Alpaca symbols list is not empty."""
        with open(alpaca_symbols_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        symbols = data.get("symbols", [])
        assert len(symbols) > 0, "Alpaca symbols list is empty"

    def test_crypto_symbols_are_usdt_pairs(self, crypto_symbols_path: Path):
        """Test crypto symbols are USDT pairs."""
        with open(crypto_symbols_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for symbol in data:
            assert symbol.endswith("USDT"), f"{symbol} is not a USDT pair"

    def test_alpaca_symbols_are_uppercase(self, alpaca_symbols_path: Path):
        """Test Alpaca symbols are uppercase."""
        with open(alpaca_symbols_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        symbols = data.get("symbols", [])
        for symbol in symbols:
            assert symbol == symbol.upper(), f"{symbol} is not uppercase"

    def test_crypto_contains_major_coins(self, crypto_symbols_path: Path):
        """Test crypto symbols contain major coins."""
        with open(crypto_symbols_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "BTCUSDT" in data, "BTCUSDT missing from crypto symbols"
        assert "ETHUSDT" in data, "ETHUSDT missing from crypto symbols"

    def test_alpaca_contains_major_stocks(self, alpaca_symbols_path: Path):
        """Test Alpaca symbols contain major stocks."""
        with open(alpaca_symbols_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        symbols = data.get("symbols", [])
        assert "AAPL" in symbols, "AAPL missing from Alpaca symbols"
        assert "MSFT" in symbols, "MSFT missing from Alpaca symbols"
        assert "SPY" in symbols, "SPY ETF missing from Alpaca symbols"

    def test_alpaca_contains_gold_etfs(self, alpaca_symbols_path: Path):
        """Test Alpaca symbols contain gold ETFs."""
        with open(alpaca_symbols_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        symbols = data.get("symbols", [])
        assert "GLD" in symbols, "GLD missing from Alpaca symbols"
        assert "IAU" in symbols, "IAU missing from Alpaca symbols"

    def test_alpaca_has_metadata(self, alpaca_symbols_path: Path):
        """Test Alpaca symbols file has metadata."""
        with open(alpaca_symbols_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "vendor" in data
        assert "asset_class" in data
        assert "count" in data
        assert data["vendor"] == "alpaca"
        assert data["asset_class"] == "us_equity"
        assert data["count"] == len(data.get("symbols", []))


# =============================================================================
# Test Backward Compatibility
# =============================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing crypto configs."""

    def test_crypto_config_still_loads(self, signal_only_crypto_config_path: Path):
        """Test crypto config can still be loaded."""
        with open(signal_only_crypto_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        assert config is not None
        assert config.get("mode") == "train"

    def test_crypto_config_unchanged_asset_class(self, signal_only_crypto_config: Dict[str, Any]):
        """Test crypto config still has crypto asset class."""
        assert signal_only_crypto_config["asset_class"] == "crypto"

    def test_crypto_config_unchanged_vendor(self, signal_only_crypto_config: Dict[str, Any]):
        """Test crypto config still has binance vendor."""
        assert signal_only_crypto_config["data_vendor"] == "binance"

    def test_crypto_config_unchanged_tif(self, signal_only_crypto_config: Dict[str, Any]):
        """Test crypto config still has GTC TIF."""
        exec_params = signal_only_crypto_config.get("execution_params", {})
        assert exec_params.get("tif") == "GTC"

    def test_stocks_training_config_still_loads(self, stocks_config_path: Path):
        """Test stocks training config can still be loaded."""
        with open(stocks_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        assert config is not None
        assert config.get("mode") == "train"

    def test_stocks_training_config_has_equity(self, stocks_config: Dict[str, Any]):
        """Test stocks training config has equity asset class."""
        assert stocks_config["asset_class"] == "equity"


# =============================================================================
# Test Config Schema Validity
# =============================================================================


class TestConfigSchemaValidity:
    """Tests for config schema validity."""

    def test_stocks_config_has_required_sections(self, signal_only_stocks_config: Dict[str, Any]):
        """Test stocks config has all required sections."""
        required_sections = [
            "mode",
            "asset_class",
            "data",
            "execution",
            "env",
            "model",
            "fees",
            "slippage",
            "training",
        ]
        for section in required_sections:
            assert section in signal_only_stocks_config, f"Missing required section: {section}"

    def test_model_params_has_required_fields(self, signal_only_stocks_config: Dict[str, Any]):
        """Test model.params has required fields."""
        params = signal_only_stocks_config.get("model", {}).get("params", {})
        required_fields = [
            "learning_rate",
            "gamma",
            "gae_lambda",
            "clip_range",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
        ]
        for field in required_fields:
            assert field in params, f"Missing required model param: {field}"

    def test_data_has_required_fields(self, signal_only_stocks_config: Dict[str, Any]):
        """Test data section has required fields."""
        data = signal_only_stocks_config.get("data", {})
        required_fields = ["timeframe", "train_start_ts", "train_end_ts"]
        for field in required_fields:
            assert field in data, f"Missing required data field: {field}"

    def test_training_has_required_fields(self, signal_only_stocks_config: Dict[str, Any]):
        """Test training section has required fields."""
        training = signal_only_stocks_config.get("training", {})
        required_fields = ["total_timesteps", "n_envs"]
        for field in required_fields:
            assert field in training, f"Missing required training field: {field}"


# =============================================================================
# Test Universe Loading Integration
# =============================================================================


class TestUniverseLoadingIntegration:
    """Integration tests for universe loading."""

    def test_crypto_universe_can_be_loaded_as_list(self, crypto_symbols_path: Path):
        """Test crypto universe can be loaded as a simple list."""
        with open(crypto_symbols_path, "r", encoding="utf-8") as f:
            symbols = json.load(f)

        assert isinstance(symbols, list)
        assert all(isinstance(s, str) for s in symbols)

    def test_alpaca_universe_symbols_can_be_extracted(self, alpaca_symbols_path: Path):
        """Test Alpaca universe symbols can be extracted from the dict."""
        with open(alpaca_symbols_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        symbols = data.get("symbols", [])
        assert isinstance(symbols, list)
        assert all(isinstance(s, str) for s in symbols)

    def test_universes_have_no_overlap(self, crypto_symbols_path: Path, alpaca_symbols_path: Path):
        """Test crypto and stock universes don't overlap."""
        with open(crypto_symbols_path, "r", encoding="utf-8") as f:
            crypto_symbols = set(json.load(f))

        with open(alpaca_symbols_path, "r", encoding="utf-8") as f:
            stock_symbols = set(json.load(f).get("symbols", []))

        overlap = crypto_symbols & stock_symbols
        assert len(overlap) == 0, f"Unexpected overlap: {overlap}"
