# -*- coding: utf-8 -*-
"""
adapters/binance/__init__.py
Binance adapter implementations.

This package provides adapter implementations for Binance exchange,
wrapping existing functionality from binance_public.py, impl_fees.py, etc.

Usage:
    from adapters.binance import (
        BinanceMarketDataAdapter,
        BinanceFeeAdapter,
        BinanceTradingHoursAdapter,
        BinanceExchangeInfoAdapter,
        # Futures adapters
        BinanceFuturesMarketDataAdapter,
        BinanceFuturesExchangeInfoAdapter,
        BinanceFuturesOrderExecutionAdapter,
    )

    # Or use the registry
    from adapters.registry import create_market_data_adapter
    adapter = create_market_data_adapter("binance", config)

    # For futures
    from adapters.registry import create_futures_market_data_adapter
    futures_adapter = create_futures_market_data_adapter("binance_futures", config)
"""

from __future__ import annotations

import logging

from adapters.models import ExchangeVendor
from adapters.registry import AdapterType, register

logger = logging.getLogger(__name__)

# Import adapter implementations
from .market_data import BinanceMarketDataAdapter
from .fees import BinanceFeeAdapter
from .trading_hours import BinanceTradingHoursAdapter
from .exchange_info import BinanceExchangeInfoAdapter
from .order_execution import BinanceOrderExecutionAdapter

# Import futures adapter implementations
from .futures_market_data import BinanceFuturesMarketDataAdapter
from .futures_exchange_info import BinanceFuturesExchangeInfoAdapter
from .futures_order_execution import BinanceFuturesOrderExecutionAdapter


# Register adapters with the global registry
def _register_adapters() -> None:
    """Register Binance adapters with the global registry."""

    # Market Data
    register(
        vendor=ExchangeVendor.BINANCE,
        adapter_type=AdapterType.MARKET_DATA,
        adapter_class=BinanceMarketDataAdapter,
        description="Binance market data adapter (REST/WebSocket)",
    )

    # Fees
    register(
        vendor=ExchangeVendor.BINANCE,
        adapter_type=AdapterType.FEE,
        adapter_class=BinanceFeeAdapter,
        description="Binance fee computation adapter",
    )

    # Trading Hours
    register(
        vendor=ExchangeVendor.BINANCE,
        adapter_type=AdapterType.TRADING_HOURS,
        adapter_class=BinanceTradingHoursAdapter,
        description="Binance trading hours (24/7 crypto)",
    )

    # Exchange Info
    register(
        vendor=ExchangeVendor.BINANCE,
        adapter_type=AdapterType.EXCHANGE_INFO,
        adapter_class=BinanceExchangeInfoAdapter,
        description="Binance exchange info adapter",
    )

    # Spot Order Execution (P0-C: closes the crypto-spot live/panic gap)
    register(
        vendor=ExchangeVendor.BINANCE,
        adapter_type=AdapterType.ORDER_EXECUTION,
        adapter_class=BinanceOrderExecutionAdapter,
        description="Binance Spot order execution (submit/cancel/positions)",
    )

    # Also register for Binance US (same implementations)
    register(
        vendor=ExchangeVendor.BINANCE_US,
        adapter_type=AdapterType.MARKET_DATA,
        adapter_class=BinanceMarketDataAdapter,
        default_config={"base_url": "https://api.binance.us"},
        description="Binance US market data adapter",
    )

    register(
        vendor=ExchangeVendor.BINANCE_US,
        adapter_type=AdapterType.FEE,
        adapter_class=BinanceFeeAdapter,
        description="Binance US fee computation adapter",
    )

    register(
        vendor=ExchangeVendor.BINANCE_US,
        adapter_type=AdapterType.TRADING_HOURS,
        adapter_class=BinanceTradingHoursAdapter,
        description="Binance US trading hours (24/7 crypto)",
    )

    register(
        vendor=ExchangeVendor.BINANCE_US,
        adapter_type=AdapterType.EXCHANGE_INFO,
        adapter_class=BinanceExchangeInfoAdapter,
        default_config={"base_url": "https://api.binance.us"},
        description="Binance US exchange info adapter",
    )

    register(
        vendor=ExchangeVendor.BINANCE_US,
        adapter_type=AdapterType.ORDER_EXECUTION,
        adapter_class=BinanceOrderExecutionAdapter,
        default_config={"spot_url": "https://api.binance.us"},
        description="Binance US spot order execution",
    )

    # ===========================
    # Futures Adapters
    # ===========================

    # Binance USDT-M Futures
    register(
        vendor=ExchangeVendor.BINANCE_FUTURES,
        adapter_type=AdapterType.FUTURES_MARKET_DATA,
        adapter_class=BinanceFuturesMarketDataAdapter,
        description="Binance Futures market data (mark price, funding, OI)",
    )

    register(
        vendor=ExchangeVendor.BINANCE_FUTURES,
        adapter_type=AdapterType.FUTURES_EXCHANGE_INFO,
        adapter_class=BinanceFuturesExchangeInfoAdapter,
        description="Binance Futures exchange info (contracts, leverage)",
    )

    register(
        vendor=ExchangeVendor.BINANCE_FUTURES,
        adapter_type=AdapterType.FUTURES_ORDER_EXECUTION,
        adapter_class=BinanceFuturesOrderExecutionAdapter,
        description="Binance Futures order execution (margin, leverage)",
    )

    # Also register futures adapters for standard BINANCE vendor
    # (allows using "binance" with futures adapter types)
    register(
        vendor=ExchangeVendor.BINANCE,
        adapter_type=AdapterType.FUTURES_MARKET_DATA,
        adapter_class=BinanceFuturesMarketDataAdapter,
        description="Binance Futures market data (via BINANCE vendor)",
    )

    register(
        vendor=ExchangeVendor.BINANCE,
        adapter_type=AdapterType.FUTURES_EXCHANGE_INFO,
        adapter_class=BinanceFuturesExchangeInfoAdapter,
        description="Binance Futures exchange info (via BINANCE vendor)",
    )

    register(
        vendor=ExchangeVendor.BINANCE,
        adapter_type=AdapterType.FUTURES_ORDER_EXECUTION,
        adapter_class=BinanceFuturesOrderExecutionAdapter,
        description="Binance Futures order execution (via BINANCE vendor)",
    )

    logger.debug("Registered Binance adapters (including futures)")


# Auto-register on import
_register_adapters()


__all__ = [
    # Spot adapters
    "BinanceMarketDataAdapter",
    "BinanceFeeAdapter",
    "BinanceTradingHoursAdapter",
    "BinanceExchangeInfoAdapter",
    "BinanceOrderExecutionAdapter",
    # Futures adapters
    "BinanceFuturesMarketDataAdapter",
    "BinanceFuturesExchangeInfoAdapter",
    "BinanceFuturesOrderExecutionAdapter",
]
