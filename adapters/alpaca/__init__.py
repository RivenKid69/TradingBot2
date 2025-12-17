# -*- coding: utf-8 -*-
"""
adapters/alpaca/__init__.py
Alpaca adapter implementations for US equity trading.

This package provides adapter implementations for Alpaca Markets,
enabling stock trading with the same interface as crypto trading.

Status: Tested and operational (Phase 2 Complete)

Usage:
    from adapters.alpaca import (
        AlpacaMarketDataAdapter,
        AlpacaFeeAdapter,
        AlpacaTradingHoursAdapter,
        AlpacaExchangeInfoAdapter,
        AlpacaOrderExecutionAdapter,
    )

    # Or use the registry
    from adapters.registry import create_market_data_adapter
    adapter = create_market_data_adapter("alpaca", config)

Requirements:
    pip install alpaca-py  # Alpaca official Python SDK
"""

from __future__ import annotations

import logging

from adapters.models import ExchangeVendor
from adapters.registry import AdapterType, register

logger = logging.getLogger(__name__)

# Import adapter implementations
from .market_data import AlpacaMarketDataAdapter
from .fees import AlpacaFeeAdapter
from .trading_hours import AlpacaTradingHoursAdapter
from .exchange_info import AlpacaExchangeInfoAdapter
from .order_execution import AlpacaOrderExecutionAdapter
from .options_execution import (
    AlpacaOptionsExecutionAdapter,
    OptionContract,
    OptionChain,
    OptionOrderConfig,
    OptionOrderResult,
    OptionType,
    OptionStrategy,
    create_options_adapter,
    parse_occ_symbol,
)


# Register adapters with the global registry
def _register_adapters() -> None:
    """Register Alpaca adapters with the global registry."""

    # Market Data
    register(
        vendor=ExchangeVendor.ALPACA,
        adapter_type=AdapterType.MARKET_DATA,
        adapter_class=AlpacaMarketDataAdapter,
        description="Alpaca market data adapter (REST/WebSocket)",
    )

    # Fees
    register(
        vendor=ExchangeVendor.ALPACA,
        adapter_type=AdapterType.FEE,
        adapter_class=AlpacaFeeAdapter,
        description="Alpaca fee computation (commission-free stocks)",
    )

    # Trading Hours
    register(
        vendor=ExchangeVendor.ALPACA,
        adapter_type=AdapterType.TRADING_HOURS,
        adapter_class=AlpacaTradingHoursAdapter,
        description="US equity market hours (NYSE/NASDAQ)",
    )

    # Exchange Info
    register(
        vendor=ExchangeVendor.ALPACA,
        adapter_type=AdapterType.EXCHANGE_INFO,
        adapter_class=AlpacaExchangeInfoAdapter,
        description="Alpaca exchange info adapter",
    )

    # Order Execution
    register(
        vendor=ExchangeVendor.ALPACA,
        adapter_type=AdapterType.ORDER_EXECUTION,
        adapter_class=AlpacaOrderExecutionAdapter,
        description="Alpaca order execution adapter",
    )

    logger.debug("Registered Alpaca adapters")


# Auto-register on import
_register_adapters()


__all__ = [
    "AlpacaMarketDataAdapter",
    "AlpacaFeeAdapter",
    "AlpacaTradingHoursAdapter",
    "AlpacaExchangeInfoAdapter",
    "AlpacaOrderExecutionAdapter",
    # Options Trading (Phase 6)
    "AlpacaOptionsExecutionAdapter",
    "OptionContract",
    "OptionChain",
    "OptionOrderConfig",
    "OptionOrderResult",
    "OptionType",
    "OptionStrategy",
    "create_options_adapter",
    "parse_occ_symbol",
]
