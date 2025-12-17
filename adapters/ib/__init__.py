# -*- coding: utf-8 -*-
"""
adapters/ib/__init__.py
Interactive Brokers adapter implementations for CME/CBOT/NYMEX/COMEX futures.

This package provides adapter implementations for Interactive Brokers TWS API,
enabling futures trading with ES, NQ, GC, CL, 6E and other CME Group contracts.

Status: Tested and operational (Phase 3B Complete)

Key Features:
- Real-time market data via TWS/Gateway
- Historical bars with pacing compliance
- Order execution (market, limit, stop, bracket)
- Position and margin queries
- Contract specifications and rollover info

Rate Limits (IB TWS API):
- General messages: 50/sec (we use 45 for safety)
- Historical data: 60 requests per 10 minutes
- Identical requests: 6 per 10 minutes
- Market data lines: 100 concurrent

Usage:
    from adapters.ib import (
        IBMarketDataAdapter,
        IBOrderExecutionAdapter,
        IBExchangeInfoAdapter,
        IBConnectionManager,
        IBRateLimiter,
    )

    # Or use the registry
    from adapters.registry import create_market_data_adapter
    adapter = create_market_data_adapter("ib", config)

Requirements:
    pip install ib_insync  # IB TWS API wrapper

References:
    - IB TWS API: https://interactivebrokers.github.io/tws-api/
    - ib_insync: https://ib-insync.readthedocs.io/
    - CME Contract Specs: https://www.cmegroup.com/trading/products/
"""

from __future__ import annotations

import logging

from adapters.models import ExchangeVendor
from adapters.registry import AdapterType, register

logger = logging.getLogger(__name__)

# Import adapter implementations
from .market_data import (
    IBMarketDataAdapter,
    IBConnectionManager,
    IBRateLimiter,
)
from .order_execution import IBOrderExecutionAdapter
from .exchange_info import IBExchangeInfoAdapter


# Register adapters with the global registry
def _register_adapters() -> None:
    """Register IB adapters with the global registry."""

    # IB Generic (auto-routes based on symbol)
    register(
        vendor=ExchangeVendor.IB,
        adapter_type=AdapterType.FUTURES_MARKET_DATA,
        adapter_class=IBMarketDataAdapter,
        description="IB TWS market data for CME Group futures",
    )

    register(
        vendor=ExchangeVendor.IB,
        adapter_type=AdapterType.FUTURES_ORDER_EXECUTION,
        adapter_class=IBOrderExecutionAdapter,
        description="IB TWS order execution for CME Group futures",
    )

    register(
        vendor=ExchangeVendor.IB,
        adapter_type=AdapterType.FUTURES_EXCHANGE_INFO,
        adapter_class=IBExchangeInfoAdapter,
        description="IB contract specifications and info",
    )

    # IB CME (equity index, currencies)
    register(
        vendor=ExchangeVendor.IB_CME,
        adapter_type=AdapterType.FUTURES_MARKET_DATA,
        adapter_class=IBMarketDataAdapter,
        default_config={"default_exchange": "CME"},
        description="IB TWS market data for CME (ES, NQ, 6E, 6J)",
    )

    register(
        vendor=ExchangeVendor.IB_CME,
        adapter_type=AdapterType.FUTURES_ORDER_EXECUTION,
        adapter_class=IBOrderExecutionAdapter,
        default_config={"default_exchange": "CME"},
        description="IB TWS order execution for CME",
    )

    register(
        vendor=ExchangeVendor.IB_CME,
        adapter_type=AdapterType.FUTURES_EXCHANGE_INFO,
        adapter_class=IBExchangeInfoAdapter,
        default_config={"default_exchange": "CME"},
        description="IB contract info for CME",
    )

    # IB CBOT (bonds, grains)
    register(
        vendor=ExchangeVendor.IB_CBOT,
        adapter_type=AdapterType.FUTURES_MARKET_DATA,
        adapter_class=IBMarketDataAdapter,
        default_config={"default_exchange": "CBOT"},
        description="IB TWS market data for CBOT (ZN, ZB, ZC)",
    )

    register(
        vendor=ExchangeVendor.IB_CBOT,
        adapter_type=AdapterType.FUTURES_ORDER_EXECUTION,
        adapter_class=IBOrderExecutionAdapter,
        default_config={"default_exchange": "CBOT"},
        description="IB TWS order execution for CBOT",
    )

    register(
        vendor=ExchangeVendor.IB_CBOT,
        adapter_type=AdapterType.FUTURES_EXCHANGE_INFO,
        adapter_class=IBExchangeInfoAdapter,
        default_config={"default_exchange": "CBOT"},
        description="IB contract info for CBOT",
    )

    # IB NYMEX (energy)
    register(
        vendor=ExchangeVendor.IB_NYMEX,
        adapter_type=AdapterType.FUTURES_MARKET_DATA,
        adapter_class=IBMarketDataAdapter,
        default_config={"default_exchange": "NYMEX"},
        description="IB TWS market data for NYMEX (CL, NG)",
    )

    register(
        vendor=ExchangeVendor.IB_NYMEX,
        adapter_type=AdapterType.FUTURES_ORDER_EXECUTION,
        adapter_class=IBOrderExecutionAdapter,
        default_config={"default_exchange": "NYMEX"},
        description="IB TWS order execution for NYMEX",
    )

    register(
        vendor=ExchangeVendor.IB_NYMEX,
        adapter_type=AdapterType.FUTURES_EXCHANGE_INFO,
        adapter_class=IBExchangeInfoAdapter,
        default_config={"default_exchange": "NYMEX"},
        description="IB contract info for NYMEX",
    )

    # IB COMEX (metals)
    register(
        vendor=ExchangeVendor.IB_COMEX,
        adapter_type=AdapterType.FUTURES_MARKET_DATA,
        adapter_class=IBMarketDataAdapter,
        default_config={"default_exchange": "COMEX"},
        description="IB TWS market data for COMEX (GC, SI)",
    )

    register(
        vendor=ExchangeVendor.IB_COMEX,
        adapter_type=AdapterType.FUTURES_ORDER_EXECUTION,
        adapter_class=IBOrderExecutionAdapter,
        default_config={"default_exchange": "COMEX"},
        description="IB TWS order execution for COMEX",
    )

    register(
        vendor=ExchangeVendor.IB_COMEX,
        adapter_type=AdapterType.FUTURES_EXCHANGE_INFO,
        adapter_class=IBExchangeInfoAdapter,
        default_config={"default_exchange": "COMEX"},
        description="IB contract info for COMEX",
    )

    logger.debug("Registered IB adapters for CME Group futures")


# Auto-register on import
_register_adapters()


__all__ = [
    # Core adapters
    "IBMarketDataAdapter",
    "IBOrderExecutionAdapter",
    "IBExchangeInfoAdapter",
    # Connection management
    "IBConnectionManager",
    "IBRateLimiter",
]
