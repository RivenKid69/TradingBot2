# -*- coding: utf-8 -*-
"""
Dukascopy Forex Adapter Package.

Provides forex/metals/CFD market data via Dukascopy's **free public historical
tick feed** (hourly LZMA-compressed ``.bi5`` files) — no credentials required,
matching the UI's "Dukascopy (Public ticks)" offering. Data-only: order
execution via the public feed is not possible, so only the market-data adapter
is registered (like Yahoo/Polygon).

Usage:
    from adapters.registry import create_market_data_adapter
    adapter = create_market_data_adapter("dukascopy")
    bars = adapter.get_bars("EURUSD", "1h", limit=24)

References:
    - Historical data: https://www.dukascopy.com/swiss/english/marketwatch/historical/
    - bi5 format: community-documented (duka / dukascopy-node / dukascopy-python)
"""

from __future__ import annotations

import logging

from adapters.models import ExchangeVendor
from adapters.registry import AdapterType, register

from .market_data import DukascopyMarketDataAdapter

logger = logging.getLogger(__name__)


def _register_adapters() -> None:
    """Register Dukascopy adapters with the global registry."""
    register(
        vendor=ExchangeVendor.DUKASCOPY,
        adapter_type=AdapterType.MARKET_DATA,
        adapter_class=DukascopyMarketDataAdapter,
        description="Dukascopy public tick feed (forex/metals) — market data only",
    )
    logger.debug("Registered Dukascopy adapters")


# Auto-register on import (registry lazy-loads this module for DUKASCOPY).
_register_adapters()


__all__ = ["DukascopyMarketDataAdapter"]
