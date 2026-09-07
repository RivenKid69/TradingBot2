# -*- coding: utf-8 -*-
"""
adapters/config.py
Configuration schema for multi-exchange support.

This module provides Pydantic models for configuring exchange adapters.
It integrates with the existing config system in core_config.py.

Usage in YAML config:
    exchange:
      vendor: "binance"  # or "alpaca"
      market_type: "CRYPTO_SPOT"  # or "EQUITY"

      # Binance-specific settings
      binance:
        base_url: "https://api.binance.com"
        use_futures: false
        timeout: 20

      # Alpaca-specific settings
      alpaca:
        api_key: "${ALPACA_API_KEY}"
        api_secret: "${ALPACA_API_SECRET}"
        paper: true
        feed: "iex"
        extended_hours: false

      # Common settings
      fees:
        include_regulatory: true

      trading_hours:
        allow_extended: true

Usage in Python:
    from adapters.config import ExchangeConfig
    config = ExchangeConfig.from_yaml("configs/exchange.yaml")
    adapter = config.create_market_data_adapter()

Validation:
    from adapters.config import validate_config
    result = validate_config(config)
    if not result.is_valid:
        for error in result.errors:
            print(f"Error: {error}")
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from urllib.parse import urlparse

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
except ImportError:
    # Fallback for older pydantic
    from pydantic import (
        BaseModel,
        Field,
        validator as field_validator,
        root_validator as model_validator,
    )

from .models import ExchangeVendor, MarketType

# Attempt to import secure logging (optional dependency)
try:
    from services.secure_logging import (
        validate_api_credentials,
        get_credential_summary,
        mask_dict,
        get_secure_logger,
    )

    _HAS_SECURE_LOGGING = True
except ImportError:
    _HAS_SECURE_LOGGING = False

    def validate_api_credentials(api_key, api_secret, **kwargs) -> Tuple[bool, Optional[str]]:
        """Fallback validation."""
        if not api_key or not api_secret:
            return False, "API credentials missing"
        return True, None

    def get_credential_summary(api_key, api_secret) -> Dict[str, Any]:
        """Fallback summary."""
        return {
            "api_key_present": bool(api_key),
            "api_secret_present": bool(api_secret),
        }

    def mask_dict(data: Dict, **kwargs) -> Dict:
        """Fallback - no masking."""
        return data

    def get_secure_logger(name: str):
        """Fallback to standard logger."""
        return logging.getLogger(name)


logger = get_secure_logger(__name__)


# =============================================================================
# Validation Result
# =============================================================================


@dataclass
class ConfigValidationResult:
    """Result of configuration validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def merge(self, other: "ConfigValidationResult") -> None:
        """Merge another validation result into this one."""
        if not other.is_valid:
            self.is_valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.update(other.info)


# =============================================================================
# Validation Constants
# =============================================================================

# Valid URL schemes
VALID_URL_SCHEMES = {"http", "https"}

# Known API base URLs for validation
KNOWN_BINANCE_URLS = {
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
    "https://testnet.binance.vision",
    "https://fapi.binance.com",
    "https://dapi.binance.com",
    "https://api.binance.us",
}

KNOWN_ALPACA_URLS = {
    "https://api.alpaca.markets",
    "https://paper-api.alpaca.markets",
    "https://data.alpaca.markets",
}

# Timeout limits
MIN_TIMEOUT_SECONDS = 1
MAX_TIMEOUT_SECONDS = 300

# Fee limits (in basis points)
MIN_FEE_BPS = 0.0
MAX_FEE_BPS = 500.0  # 5%

# VIP tier limits
MIN_VIP_TIER = 0
MAX_VIP_TIER = 9

# Valid Alpaca feeds
VALID_ALPACA_FEEDS = {"iex", "sip"}


class BinanceConfig(BaseModel):
    """Binance-specific configuration."""

    base_url: str = Field(
        default="https://api.binance.com",
        description="Binance API base URL",
    )
    futures_url: str = Field(
        default="https://fapi.binance.com",
        description="Binance Futures API URL",
    )
    use_futures: bool = Field(
        default=False,
        description="Use futures endpoints instead of spot",
    )
    timeout: int = Field(
        default=20,
        description="Request timeout in seconds",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Binance API key (for private endpoints)",
    )
    api_secret: Optional[str] = Field(
        default=None,
        description="Binance API secret (for private endpoints)",
    )

    # Fee settings
    maker_bps: float = Field(default=10.0, description="Maker fee in basis points")
    taker_bps: float = Field(default=10.0, description="Taker fee in basis points")
    use_bnb_discount: bool = Field(default=False, description="Apply BNB fee discount")
    vip_tier: int = Field(default=0, description="VIP tier level")

    # Cache paths
    filters_cache_path: Optional[str] = Field(
        default="data/binance_filters.json",
        description="Path to exchange filters cache",
    )
    fees_cache_path: Optional[str] = Field(
        default="data/fees/fees_by_symbol.json",
        description="Path to fees cache",
    )

    def to_adapter_config(self) -> Dict[str, Any]:
        """Convert to adapter configuration dict."""
        config = {
            "base_url": self.base_url,
            "futures_url": self.futures_url,
            "use_futures": self.use_futures,
            "timeout": self.timeout,
            "maker_bps": self.maker_bps,
            "taker_bps": self.taker_bps,
            "use_bnb_discount": self.use_bnb_discount,
            "vip_tier": self.vip_tier,
        }

        if self.api_key:
            config["api_key"] = self._resolve_env(self.api_key)
        if self.api_secret:
            config["api_secret"] = self._resolve_env(self.api_secret)
        if self.filters_cache_path:
            config["filters_cache_path"] = self.filters_cache_path
        if self.fees_cache_path:
            config["fee_cache_path"] = self.fees_cache_path

        return config

    @staticmethod
    def _resolve_env(value: str) -> str:
        """Resolve environment variable if value starts with $."""
        if value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.environ.get(env_var, "")
        elif value.startswith("$"):
            env_var = value[1:]
            return os.environ.get(env_var, "")
        return value


class AlpacaConfig(BaseModel):
    """Alpaca-specific configuration."""

    api_key: str = Field(
        default="",
        description="Alpaca API key (required)",
    )
    api_secret: str = Field(
        default="",
        description="Alpaca API secret (required)",
    )
    paper: bool = Field(
        default=True,
        description="Use paper trading endpoint",
    )
    feed: str = Field(
        default="iex",
        description="Data feed: 'iex' (free) or 'sip' (paid)",
    )
    extended_hours: bool = Field(
        default=False,
        description="Allow extended hours trading",
    )
    allow_extended_hours: bool = Field(
        default=True,
        description="Consider extended hours as 'market open'",
    )

    # Regulatory fees
    include_regulatory_fees: bool = Field(
        default=True,
        description="Include SEC/TAF fees in calculations",
    )
    options_per_contract_fee: float = Field(
        default=0.65,
        description="Fee per options contract",
    )

    def to_adapter_config(self) -> Dict[str, Any]:
        """Convert to adapter configuration dict."""
        return {
            "api_key": self._resolve_env(self.api_key),
            "api_secret": self._resolve_env(self.api_secret),
            "paper": self.paper,
            "feed": self.feed,
            "extended_hours": self.extended_hours,
            "allow_extended_hours": self.allow_extended_hours,
            "include_regulatory_fees": self.include_regulatory_fees,
            "options_per_contract_fee": self.options_per_contract_fee,
        }

    @staticmethod
    def _resolve_env(value: str) -> str:
        """Resolve environment variable."""
        if value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.environ.get(env_var, "")
        elif value.startswith("$"):
            env_var = value[1:]
            return os.environ.get(env_var, "")
        return value


class FeeConfig(BaseModel):
    """Common fee configuration."""

    include_regulatory: bool = Field(
        default=True,
        description="Include regulatory fees (SEC/TAF for US equities)",
    )
    custom_maker_bps: Optional[float] = Field(
        default=None,
        description="Override maker fee rate",
    )
    custom_taker_bps: Optional[float] = Field(
        default=None,
        description="Override taker fee rate",
    )


class TradingHoursConfig(BaseModel):
    """Common trading hours configuration."""

    allow_extended: bool = Field(
        default=True,
        description="Allow trading during extended hours (pre/post market)",
    )
    use_exchange_calendar: bool = Field(
        default=True,
        description="Fetch calendar from exchange API",
    )


class ExchangeConfig(BaseModel):
    """
    Main exchange configuration.

    Example YAML:
        exchange:
          vendor: "binance"
          market_type: "CRYPTO_SPOT"
          binance:
            timeout: 30
            use_bnb_discount: true
    """

    vendor: str = Field(
        default="binance",
        description="Exchange vendor: 'binance', 'alpaca', etc.",
    )
    market_type: str = Field(
        default="CRYPTO_SPOT",
        description="Market type: CRYPTO_SPOT, EQUITY, etc.",
    )

    # Vendor-specific configs
    binance: BinanceConfig = Field(default_factory=BinanceConfig)
    alpaca: AlpacaConfig = Field(default_factory=AlpacaConfig)

    # Common configs
    fees: FeeConfig = Field(default_factory=FeeConfig)
    trading_hours: TradingHoursConfig = Field(default_factory=TradingHoursConfig)

    @property
    def exchange_vendor(self) -> ExchangeVendor:
        """Get ExchangeVendor enum."""
        try:
            return ExchangeVendor(self.vendor.lower())
        except ValueError:
            return ExchangeVendor.UNKNOWN

    @property
    def exchange_market_type(self) -> MarketType:
        """Get MarketType enum."""
        try:
            return MarketType(self.market_type)
        except ValueError:
            return MarketType.CRYPTO_SPOT

    def get_adapter_config(self) -> Dict[str, Any]:
        """Get adapter configuration for current vendor."""
        vendor = self.exchange_vendor

        if vendor in (ExchangeVendor.BINANCE, ExchangeVendor.BINANCE_US):
            config = self.binance.to_adapter_config()
        elif vendor == ExchangeVendor.ALPACA:
            config = self.alpaca.to_adapter_config()
        else:
            config = {}

        # Merge common configs
        if self.fees.custom_maker_bps is not None:
            config["maker_bps"] = self.fees.custom_maker_bps
        if self.fees.custom_taker_bps is not None:
            config["taker_bps"] = self.fees.custom_taker_bps

        config["market_type"] = self.market_type

        return config

    def create_market_data_adapter(self):
        """Create market data adapter for configured vendor."""
        from .registry import create_market_data_adapter

        return create_market_data_adapter(self.vendor, self.get_adapter_config())

    def create_fee_adapter(self):
        """Create fee adapter for configured vendor."""
        from .registry import create_fee_adapter

        return create_fee_adapter(self.vendor, self.get_adapter_config())

    def create_trading_hours_adapter(self):
        """Create trading hours adapter for configured vendor."""
        from .registry import create_trading_hours_adapter

        config = self.get_adapter_config()
        config["allow_extended_hours"] = self.trading_hours.allow_extended
        config["use_alpaca_calendar"] = self.trading_hours.use_exchange_calendar
        return create_trading_hours_adapter(self.vendor, config)

    def create_exchange_info_adapter(self):
        """Create exchange info adapter for configured vendor."""
        from .registry import create_exchange_info_adapter

        return create_exchange_info_adapter(self.vendor, self.get_adapter_config())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExchangeConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExchangeConfig":
        """Load from YAML file."""
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Support both root-level and nested under 'exchange' key
        if "exchange" in data:
            data = data["exchange"]

        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()


# =========================
# Default Configuration
# =========================

DEFAULT_BINANCE_CONFIG = BinanceConfig()
DEFAULT_ALPACA_CONFIG = AlpacaConfig()
DEFAULT_EXCHANGE_CONFIG = ExchangeConfig()


def load_exchange_config(
    path: Optional[Union[str, Path]] = None,
    vendor: Optional[str] = None,
) -> ExchangeConfig:
    """
    Load exchange configuration.

    Priority:
    1. Explicit path
    2. Environment variable EXCHANGE_CONFIG_PATH
    3. Default config with optional vendor override

    Args:
        path: Path to YAML config file
        vendor: Override vendor (if not loading from file)

    Returns:
        ExchangeConfig instance
    """
    if path is not None:
        return ExchangeConfig.from_yaml(path)

    env_path = os.environ.get("EXCHANGE_CONFIG_PATH")
    if env_path and Path(env_path).exists():
        return ExchangeConfig.from_yaml(env_path)

    config = ExchangeConfig()
    if vendor:
        config.vendor = vendor

    return config


# =============================================================================
# Validation Helper Functions
# =============================================================================


def _validate_url(
    url: str,
    known_urls: Optional[set] = None,
    require_https: bool = True,
) -> Tuple[bool, Optional[str]]:
    """
    Validate a URL.

    Args:
        url: URL to validate
        known_urls: Set of known valid URLs (optional)
        require_https: Whether to require HTTPS

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url:
        return False, "URL is empty"

    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"Invalid URL format: {e}"

    # Check scheme
    if not parsed.scheme:
        return False, "URL missing scheme (http/https)"

    if parsed.scheme not in VALID_URL_SCHEMES:
        return False, f"Invalid URL scheme: {parsed.scheme}"

    if require_https and parsed.scheme != "https":
        return False, "URL must use HTTPS for security"

    # Check netloc (domain)
    if not parsed.netloc:
        return False, "URL missing domain"

    # Check against known URLs (warning only)
    if known_urls:
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        if base_url not in known_urls:
            # Return valid but this can be a warning
            pass

    return True, None


def _validate_timeout(timeout: int) -> Tuple[bool, Optional[str]]:
    """Validate timeout value."""
    if timeout < MIN_TIMEOUT_SECONDS:
        return False, f"Timeout too low: {timeout}s (min: {MIN_TIMEOUT_SECONDS}s)"
    if timeout > MAX_TIMEOUT_SECONDS:
        return False, f"Timeout too high: {timeout}s (max: {MAX_TIMEOUT_SECONDS}s)"
    return True, None


def _validate_fee_bps(fee: float, name: str) -> Tuple[bool, Optional[str]]:
    """Validate fee in basis points."""
    if fee < MIN_FEE_BPS:
        return False, f"{name} fee cannot be negative: {fee} bps"
    if fee > MAX_FEE_BPS:
        return False, f"{name} fee too high: {fee} bps (max: {MAX_FEE_BPS} bps = 5%)"
    return True, None


def _validate_vip_tier(tier: int) -> Tuple[bool, Optional[str]]:
    """Validate VIP tier."""
    if tier < MIN_VIP_TIER or tier > MAX_VIP_TIER:
        return False, f"VIP tier must be {MIN_VIP_TIER}-{MAX_VIP_TIER}, got: {tier}"
    return True, None


def _validate_alpaca_feed(feed: str) -> Tuple[bool, Optional[str]]:
    """Validate Alpaca feed type."""
    if feed.lower() not in VALID_ALPACA_FEEDS:
        return False, f"Invalid Alpaca feed: {feed}. Must be one of: {VALID_ALPACA_FEEDS}"
    return True, None


# =============================================================================
# Config Validation Functions
# =============================================================================


def validate_binance_config(config: BinanceConfig) -> ConfigValidationResult:
    """
    Validate Binance configuration.

    Args:
        config: BinanceConfig instance

    Returns:
        ConfigValidationResult with errors and warnings
    """
    result = ConfigValidationResult(is_valid=True)

    # Validate base URL
    is_valid, error = _validate_url(config.base_url, KNOWN_BINANCE_URLS)
    if not is_valid:
        result.add_error(f"base_url: {error}")
    else:
        # Check if URL is known
        parsed = urlparse(config.base_url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in KNOWN_BINANCE_URLS:
            result.add_warning(f"base_url '{config.base_url}' is not a known Binance URL")

    # Validate futures URL
    is_valid, error = _validate_url(config.futures_url, KNOWN_BINANCE_URLS)
    if not is_valid:
        result.add_error(f"futures_url: {error}")

    # Validate timeout
    is_valid, error = _validate_timeout(config.timeout)
    if not is_valid:
        result.add_error(error)

    # Validate fees
    is_valid, error = _validate_fee_bps(config.maker_bps, "Maker")
    if not is_valid:
        result.add_error(error)

    is_valid, error = _validate_fee_bps(config.taker_bps, "Taker")
    if not is_valid:
        result.add_error(error)

    # Validate VIP tier
    is_valid, error = _validate_vip_tier(config.vip_tier)
    if not is_valid:
        result.add_error(error)

    # Validate API credentials if provided
    if config.api_key or config.api_secret:
        resolved_key = config._resolve_env(config.api_key) if config.api_key else ""
        resolved_secret = config._resolve_env(config.api_secret) if config.api_secret else ""

        is_valid, error = validate_api_credentials(
            resolved_key,
            resolved_secret,
            require_both=True,
            min_key_length=32,
            min_secret_length=32,
        )
        if not is_valid:
            result.add_error(f"API credentials: {error}")

        # Add credential summary (safe, no secrets)
        result.info["binance_credentials"] = get_credential_summary(resolved_key, resolved_secret)

    # Check cache paths
    if config.filters_cache_path:
        cache_dir = Path(config.filters_cache_path).parent
        if not cache_dir.exists():
            result.add_warning(f"filters_cache_path directory does not exist: {cache_dir}")

    return result


def validate_alpaca_config(config: AlpacaConfig) -> ConfigValidationResult:
    """
    Validate Alpaca configuration.

    Args:
        config: AlpacaConfig instance

    Returns:
        ConfigValidationResult with errors and warnings
    """
    result = ConfigValidationResult(is_valid=True)

    # Validate feed
    is_valid, error = _validate_alpaca_feed(config.feed)
    if not is_valid:
        result.add_error(error)

    # Validate API credentials
    resolved_key = config._resolve_env(config.api_key)
    resolved_secret = config._resolve_env(config.api_secret)

    is_valid, error = validate_api_credentials(
        resolved_key,
        resolved_secret,
        require_both=True,
        min_key_length=16,
        min_secret_length=32,
    )
    if not is_valid:
        result.add_error(f"API credentials: {error}")

    # Add credential summary (safe, no secrets)
    result.info["alpaca_credentials"] = get_credential_summary(resolved_key, resolved_secret)

    # Warning for paper mode
    if config.paper:
        result.add_warning("Paper trading mode enabled - not suitable for live trading")

    # Check options fee
    if config.options_per_contract_fee < 0:
        result.add_error(
            f"options_per_contract_fee cannot be negative: {config.options_per_contract_fee}"
        )
    elif config.options_per_contract_fee > 5.0:
        result.add_warning(
            f"options_per_contract_fee seems high: ${config.options_per_contract_fee}"
        )

    return result


def validate_fee_config(config: FeeConfig) -> ConfigValidationResult:
    """Validate fee configuration."""
    result = ConfigValidationResult(is_valid=True)

    if config.custom_maker_bps is not None:
        is_valid, error = _validate_fee_bps(config.custom_maker_bps, "Custom maker")
        if not is_valid:
            result.add_error(error)

    if config.custom_taker_bps is not None:
        is_valid, error = _validate_fee_bps(config.custom_taker_bps, "Custom taker")
        if not is_valid:
            result.add_error(error)

    return result


def validate_config(config: ExchangeConfig) -> ConfigValidationResult:
    """
    Comprehensive validation of exchange configuration.

    Args:
        config: ExchangeConfig instance

    Returns:
        ConfigValidationResult with all errors, warnings, and info

    Example:
        >>> from adapters.config import ExchangeConfig, validate_config
        >>> config = ExchangeConfig.from_yaml("configs/exchange.yaml")
        >>> result = validate_config(config)
        >>> if not result.is_valid:
        ...     for error in result.errors:
        ...         print(f"Error: {error}")
        >>> for warning in result.warnings:
        ...     print(f"Warning: {warning}")
    """
    result = ConfigValidationResult(is_valid=True)

    # Validate vendor
    vendor = config.exchange_vendor
    if vendor == ExchangeVendor.UNKNOWN:
        result.add_error(f"Unknown vendor: '{config.vendor}'")

    # Validate market type
    try:
        MarketType(config.market_type)
    except ValueError:
        result.add_warning(f"Unknown market type: '{config.market_type}' (will use default)")

    # Validate vendor-specific config
    if vendor in (ExchangeVendor.BINANCE, ExchangeVendor.BINANCE_US):
        binance_result = validate_binance_config(config.binance)
        result.merge(binance_result)

    elif vendor == ExchangeVendor.ALPACA:
        alpaca_result = validate_alpaca_config(config.alpaca)
        result.merge(alpaca_result)

    # Validate common configs
    fee_result = validate_fee_config(config.fees)
    result.merge(fee_result)

    # Add summary info
    result.info["vendor"] = config.vendor
    result.info["market_type"] = config.market_type
    result.info["has_secure_logging"] = _HAS_SECURE_LOGGING

    logger.debug(
        "Config validation complete",
        extra={
            "is_valid": result.is_valid,
            "errors": len(result.errors),
            "warnings": len(result.warnings),
        },
    )

    return result


def validate_config_strict(config: ExchangeConfig) -> None:
    """
    Validate configuration and raise exception if invalid.

    Args:
        config: ExchangeConfig instance

    Raises:
        ValueError: If configuration is invalid

    Example:
        >>> from adapters.config import ExchangeConfig, validate_config_strict
        >>> config = ExchangeConfig.from_yaml("configs/exchange.yaml")
        >>> validate_config_strict(config)  # Raises if invalid
    """
    result = validate_config(config)
    if not result.is_valid:
        errors = "; ".join(result.errors)
        raise ValueError(f"Invalid configuration: {errors}")


def get_config_summary(config: ExchangeConfig) -> Dict[str, Any]:
    """
    Get a safe summary of configuration for logging.

    This masks sensitive values and provides a summary suitable for logs.

    Args:
        config: ExchangeConfig instance

    Returns:
        Dictionary with safe configuration summary
    """
    summary = {
        "vendor": config.vendor,
        "market_type": config.market_type,
        "trading_hours": {
            "allow_extended": config.trading_hours.allow_extended,
            "use_exchange_calendar": config.trading_hours.use_exchange_calendar,
        },
        "fees": {
            "include_regulatory": config.fees.include_regulatory,
            "has_custom_maker": config.fees.custom_maker_bps is not None,
            "has_custom_taker": config.fees.custom_taker_bps is not None,
        },
    }

    vendor = config.exchange_vendor

    if vendor in (ExchangeVendor.BINANCE, ExchangeVendor.BINANCE_US):
        summary["binance"] = {
            "base_url": config.binance.base_url,
            "use_futures": config.binance.use_futures,
            "timeout": config.binance.timeout,
            "maker_bps": config.binance.maker_bps,
            "taker_bps": config.binance.taker_bps,
            "vip_tier": config.binance.vip_tier,
            "use_bnb_discount": config.binance.use_bnb_discount,
            "has_api_key": bool(config.binance.api_key),
            "has_api_secret": bool(config.binance.api_secret),
        }

    elif vendor == ExchangeVendor.ALPACA:
        summary["alpaca"] = {
            "paper": config.alpaca.paper,
            "feed": config.alpaca.feed,
            "extended_hours": config.alpaca.extended_hours,
            "include_regulatory_fees": config.alpaca.include_regulatory_fees,
            "has_api_key": bool(config.alpaca.api_key),
            "has_api_secret": bool(config.alpaca.api_secret),
        }

    return mask_dict(summary) if _HAS_SECURE_LOGGING else summary
