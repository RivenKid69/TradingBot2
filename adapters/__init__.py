# -*- coding: utf-8 -*-
"""
adapters/__init__.py
Multi-exchange adapter framework for AI-Powered Quantitative Research Platform.

This package provides a unified interface for different exchanges (Binance, Alpaca, etc.)
and asset classes (crypto, equities) while maintaining backward compatibility with
existing code.

Architecture:
    adapters/
    ├── models.py       # Exchange-agnostic data models
    ├── base.py         # Abstract base classes (interfaces)
    ├── registry.py     # Adapter factory and registration
    ├── binance/        # Binance implementations
    │   ├── market_data.py
    │   ├── fees.py
    │   ├── trading_hours.py
    │   └── exchange_info.py
    └── alpaca/         # Alpaca implementations
        ├── market_data.py
        ├── fees.py
        ├── trading_hours.py
        ├── exchange_info.py
        └── order_execution.py

Quick Start:
    # Using the registry (recommended)
    from adapters import create_market_data_adapter, create_fee_adapter

    # Create Binance adapter
    binance_data = create_market_data_adapter("binance", config={"timeout": 30})
    bars = binance_data.get_bars("BTCUSDT", "1h", limit=100)

    # Create Alpaca adapter
    alpaca_data = create_market_data_adapter("alpaca", config={
        "api_key": "your_key",
        "api_secret": "your_secret",
    })
    bars = alpaca_data.get_bars("AAPL", "1d", limit=100)

    # Direct import
    from adapters.binance import BinanceMarketDataAdapter
    adapter = BinanceMarketDataAdapter()

Models:
    - MarketType: CRYPTO_SPOT, CRYPTO_FUTURES, EQUITY, etc.
    - ExchangeVendor: BINANCE, ALPACA, etc.
    - ExchangeRule: Trading rules (tick size, lot size, min notional)
    - TradingSession: Market hours
    - FeeSchedule: Fee structure

Interfaces:
    - MarketDataAdapter: OHLCV bars, ticks, streaming
    - FeeAdapter: Fee computation
    - TradingHoursAdapter: Market schedule
    - OrderExecutionAdapter: Order placement and management
    - ExchangeInfoAdapter: Symbol info and trading rules

See Also:
    - docs/adapters.md for full documentation
    - tests/test_adapters*.py for examples
"""

from __future__ import annotations

# =========================
# Core Models
# =========================

from .models import (
    # Enums
    MarketType,
    ExchangeVendor,
    FeeStructure,
    SessionType,
    # Exchange Rules
    ExchangeRule,
    # Trading Sessions
    TradingSession,
    MarketCalendar,
    # Fee Models
    FeeSchedule,
    # Account & Symbol Info
    AccountInfo,
    SymbolInfo,
    # Predefined
    US_EQUITY_SESSIONS,
    CRYPTO_CONTINUOUS_SESSION,
    create_us_equity_calendar,
    create_crypto_calendar,
)

# =========================
# Abstract Base Classes
# =========================

from .base import (
    # Base
    BaseAdapter,
    # Interfaces
    MarketDataAdapter,
    FeeAdapter,
    TradingHoursAdapter,
    OrderExecutionAdapter,
    ExchangeInfoAdapter,
    ExchangeAdapter,
    # Result types
    OrderResult,
    # Protocols
    SupportsMarketData,
    SupportsFees,
    SupportsTradingHours,
)

# =========================
# Registry & Factory
# =========================

from .registry import (
    # Types
    AdapterType,
    AdapterConfig,
    AdapterRegistration,
    # Registry
    AdapterRegistry,
    get_registry,
    register,
    register_adapter,
    # Factory functions
    create_market_data_adapter,
    create_fee_adapter,
    create_trading_hours_adapter,
    create_order_execution_adapter,
    create_exchange_info_adapter,
    create_exchange_adapter,
    create_from_config,
)

# =========================
# Lazy Loading of Implementations
# =========================

# Implementation packages are loaded on demand via the registry.
# Direct imports can be done when needed:
#
#   from adapters.binance import BinanceMarketDataAdapter
#   from adapters.alpaca import AlpacaMarketDataAdapter

# =========================
# Backward Compatibility
# =========================

# Re-export existing adapter
from .binance_spot_private import (
    AccountFeeInfo,
    fetch_account_fee_info,
)


# =========================
# Module-level convenience
# =========================


def get_supported_vendors() -> list:
    """Get list of supported exchange vendors."""
    return get_registry().get_supported_vendors()


def get_supported_adapter_types(vendor: str) -> list:
    """Get adapter types supported by vendor."""
    try:
        v = ExchangeVendor(vendor.lower())
    except ValueError:
        return []
    return get_registry().get_supported_types(v)


# =========================
# Public API
# =========================

__all__ = [
    # Models - Enums
    "MarketType",
    "ExchangeVendor",
    "FeeStructure",
    "SessionType",
    # Models - Data
    "ExchangeRule",
    "TradingSession",
    "MarketCalendar",
    "FeeSchedule",
    "AccountInfo",
    "SymbolInfo",
    # Models - Predefined
    "US_EQUITY_SESSIONS",
    "CRYPTO_CONTINUOUS_SESSION",
    "create_us_equity_calendar",
    "create_crypto_calendar",
    # Base Classes
    "BaseAdapter",
    "MarketDataAdapter",
    "FeeAdapter",
    "TradingHoursAdapter",
    "OrderExecutionAdapter",
    "ExchangeInfoAdapter",
    "ExchangeAdapter",
    "OrderResult",
    # Protocols
    "SupportsMarketData",
    "SupportsFees",
    "SupportsTradingHours",
    # Registry
    "AdapterType",
    "AdapterConfig",
    "AdapterRegistration",
    "AdapterRegistry",
    "get_registry",
    "register",
    "register_adapter",
    # Factory Functions
    "create_market_data_adapter",
    "create_fee_adapter",
    "create_trading_hours_adapter",
    "create_order_execution_adapter",
    "create_exchange_info_adapter",
    "create_exchange_adapter",
    "create_from_config",
    # Backward Compatibility
    "AccountFeeInfo",
    "fetch_account_fee_info",
    # Convenience
    "get_supported_vendors",
    "get_supported_adapter_types",
]
