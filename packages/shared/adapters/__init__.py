# -*- coding: utf-8 -*-
"""
CCEA Shared Adapters - Data-Only Components.

This module contains ONLY data-access adapters that are safe for
both Cloud and Agent zones:

- MarketDataAdapter: Public market data (OHLCV, ticks, orderbook)
- FeeAdapter: Fee calculations and schedules
- TradingHoursAdapter: Market hours and calendars
- ExchangeInfoAdapter: Symbol info, trading rules

IMPORTANT: This module contains NO order execution code.
Order execution adapters are in packages/agent/execution/ ONLY.

Zone Classification:
- These adapters: SHARED (safe for Cloud and Agent)
- Order execution: AGENT ONLY (never in Cloud)
"""

from typing import Final, List

# Zone classification
ZONE: Final[str] = "shared"

# What this module provides (data-only)
DATA_ONLY_ADAPTERS: Final[List[str]] = [
    "MarketDataAdapter",
    "FeeAdapter",
    "TradingHoursAdapter",
    "ExchangeInfoAdapter",
]

# What this module explicitly DOES NOT provide
# (these are AGENT-only and must never be here)
PROHIBITED_IN_THIS_MODULE: Final[List[str]] = [
    "OrderExecutionAdapter",
    "OptionsExecutionAdapter",
    "BrokerConnector",
    "PrivateTradingClient",
]

# Re-export from existing adapters (data-only parts)
from adapters.base import (
    MarketDataAdapter,
    FeeAdapter,
    TradingHoursAdapter,
    ExchangeInfoAdapter,
    BaseAdapter,
)

from adapters.models import (
    MarketType,
    ExchangeVendor,
    FeeStructure,
    SessionType,
    ExchangeRule,
    TradingSession,
    MarketCalendar,
    FeeSchedule,
    SymbolInfo,
)

__all__ = [
    # Base classes
    "BaseAdapter",
    # Data-only adapters
    "MarketDataAdapter",
    "FeeAdapter",
    "TradingHoursAdapter",
    "ExchangeInfoAdapter",
    # Models
    "MarketType",
    "ExchangeVendor",
    "FeeStructure",
    "SessionType",
    "ExchangeRule",
    "TradingSession",
    "MarketCalendar",
    "FeeSchedule",
    "SymbolInfo",
    # Zone info
    "ZONE",
    "DATA_ONLY_ADAPTERS",
    "PROHIBITED_IN_THIS_MODULE",
]
