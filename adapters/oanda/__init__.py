# -*- coding: utf-8 -*-
"""
OANDA Forex Adapter Package

Status: Tested and operational (Phase 2 Complete)

This package provides adapter implementations for OANDA Markets,
enabling forex trading with the same interface as crypto and stock trading.

Supported Features:
- OANDA v20 REST API integration
- Streaming prices via HTTP streaming
- Historical candles (S5 to M granularity)
- Order execution with last-look handling
- Swap rate queries
- Position management
- Account information

API Rate Limits:
- 120 requests per second (streaming is separate)
- Max 5000 candles per request
- Burst: 200 requests allowed

Environment Variables:
- OANDA_API_KEY: API access token
- OANDA_ACCOUNT_ID: Trading account ID
- OANDA_PRACTICE: "true" for demo, "false" for live

Usage:
    from adapters.oanda import (
        OandaMarketDataAdapter,
        OandaFeeAdapter,
        OandaTradingHoursAdapter,
        OandaExchangeInfoAdapter,
        OandaOrderExecutionAdapter,
    )

    # Or use the registry
    from adapters.registry import create_market_data_adapter
    adapter = create_market_data_adapter("oanda", config)

Requirements:
    pip install requests aiohttp  # For API calls and async streaming

References:
    - OANDA v20 API: https://developer.oanda.com/rest-live-v20/introduction/
    - BIS Triennial Survey 2022: https://www.bis.org/statistics/rpfx22.htm
"""

from __future__ import annotations

import logging

from adapters.models import ExchangeVendor
from adapters.registry import AdapterType, register

logger = logging.getLogger(__name__)

# Import adapter implementations
from .market_data import OandaMarketDataAdapter
from .fees import OandaFeeAdapter
from .trading_hours import OandaTradingHoursAdapter
from .exchange_info import OandaExchangeInfoAdapter
from .order_execution import OandaOrderExecutionAdapter


# Register adapters with the global registry
def _register_adapters() -> None:
    """Register OANDA adapters with the global registry."""

    # Market Data
    register(
        vendor=ExchangeVendor.OANDA,
        adapter_type=AdapterType.MARKET_DATA,
        adapter_class=OandaMarketDataAdapter,
        description="OANDA forex market data adapter (REST/HTTP streaming)",
    )

    # Fees
    register(
        vendor=ExchangeVendor.OANDA,
        adapter_type=AdapterType.FEE,
        adapter_class=OandaFeeAdapter,
        description="OANDA fee computation (spread-based, no commission)",
    )

    # Trading Hours
    register(
        vendor=ExchangeVendor.OANDA,
        adapter_type=AdapterType.TRADING_HOURS,
        adapter_class=OandaTradingHoursAdapter,
        description="Forex market hours (Sun 5pm - Fri 5pm ET)",
    )

    # Exchange Info
    register(
        vendor=ExchangeVendor.OANDA,
        adapter_type=AdapterType.EXCHANGE_INFO,
        adapter_class=OandaExchangeInfoAdapter,
        description="OANDA exchange info adapter (currency pairs)",
    )

    # Order Execution
    register(
        vendor=ExchangeVendor.OANDA,
        adapter_type=AdapterType.ORDER_EXECUTION,
        adapter_class=OandaOrderExecutionAdapter,
        description="OANDA order execution adapter",
    )

    logger.debug("Registered OANDA adapters")


# Auto-register on import
_register_adapters()


__all__ = [
    "OandaMarketDataAdapter",
    "OandaFeeAdapter",
    "OandaTradingHoursAdapter",
    "OandaExchangeInfoAdapter",
    "OandaOrderExecutionAdapter",
]
