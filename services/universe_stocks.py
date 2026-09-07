# -*- coding: utf-8 -*-
"""
services/universe_stocks.py
Service for managing and auto-updating stock universe.

Provides:
- Automatic stock list updates from Alpaca
- TTL-based caching for performance
- Filtering by exchange, liquidity, etc.
- Popular/liquid stock presets
- Backward compatibility with crypto universe API

Architecture:
    Alpaca API → Universe Service → Config/Pipeline

Usage:
    from services.universe_stocks import (
        get_symbols,
        get_popular_symbols,
        refresh_universe,
    )

    # Get all tradable symbols (auto-refreshes if stale)
    symbols = get_symbols()

    # Get popular/liquid stocks only
    popular = get_popular_symbols()

    # Force refresh
    symbols = get_symbols(force=True)

    # Get symbols with metadata
    assets = get_assets(exchange="NYSE")

Environment Variables:
    ALPACA_API_KEY: Alpaca API key
    ALPACA_API_SECRET: Alpaca API secret

Best Practices:
    - Use TTL-based caching to avoid API rate limits
    - Pre-filter by tradability and liquidity
    - Handle network errors gracefully

Author: Sultan Suyunbayev
Date: 2025-11-28
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default paths
DEFAULT_UNIVERSE_PATH = "data/universe/stock_symbols.json"
DEFAULT_POPULAR_PATH = "data/universe/stock_popular.json"
DEFAULT_CACHE_DIR = Path("data/universe")

# Default TTL (24 hours)
DEFAULT_TTL_SECONDS = 24 * 60 * 60

# Popular/liquid stocks (S&P 500 components + popular retail stocks)
POPULAR_SYMBOLS: Set[str] = {
    # Tech giants
    "AAPL",
    "MSFT",
    "GOOGL",
    "GOOG",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "AMD",
    "INTC",
    "CRM",
    "ORCL",
    "ADBE",
    "CSCO",
    "AVGO",
    "TXN",
    "QCOM",
    "ASML",
    "AMAT",
    "MU",
    # Finance
    "JPM",
    "BAC",
    "WFC",
    "GS",
    "MS",
    "C",
    "BRK.B",
    "V",
    "MA",
    "AXP",
    "BLK",
    "SCHW",
    "USB",
    "PNC",
    "TFC",
    # Healthcare
    "JNJ",
    "UNH",
    "PFE",
    "ABBV",
    "MRK",
    "LLY",
    "TMO",
    "ABT",
    "BMY",
    "AMGN",
    "GILD",
    "CVS",
    "CI",
    "HCA",
    # Consumer
    "WMT",
    "COST",
    "HD",
    "MCD",
    "NKE",
    "SBUX",
    "TGT",
    "LOW",
    "DIS",
    "NFLX",
    "PG",
    "KO",
    "PEP",
    "PM",
    "MO",
    # Energy
    "XOM",
    "CVX",
    "COP",
    "SLB",
    "EOG",
    "OXY",
    "MPC",
    "VLO",
    "PSX",
    "KMI",
    # Industrial
    "CAT",
    "DE",
    "HON",
    "UNP",
    "UPS",
    "FDX",
    "BA",
    "LMT",
    "RTX",
    "GE",
    # ETFs
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "VTI",
    "VOO",
    "VXX",
    "ARKK",
    "XLF",
    "XLE",
    "XLK",
    "XLV",
    "XLI",
    "XLC",
    "XLY",
    "XLP",
    "XLB",
    "XLU",
    "XLRE",
    # Precious Metals ETFs
    "GLD",
    "IAU",
    "SGOL",
    "SLV",
    # Bond ETFs
    "TLT",
    "IEF",
    "SHY",
    "BND",
    "AGG",
}

# Highly liquid stocks (subset of popular)
HIGHLY_LIQUID_SYMBOLS: Set[str] = {
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "SPY",
    "QQQ",
    "IWM",
    "VTI",
    "VOO",
    "JPM",
    "BAC",
    "V",
    "MA",
}


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class StockUniverseConfig:
    """Configuration for stock universe service."""

    cache_dir: Path = DEFAULT_CACHE_DIR
    symbols_path: str = DEFAULT_UNIVERSE_PATH
    popular_path: str = DEFAULT_POPULAR_PATH
    ttl_seconds: int = DEFAULT_TTL_SECONDS
    paper: bool = True  # Use paper trading endpoint
    default_exchange: Optional[str] = None  # Filter by exchange
    fractionable_only: bool = False  # Only fractionable stocks
    shortable_only: bool = False  # Only shortable stocks
    include_etfs: bool = True  # Include ETFs


# =============================================================================
# Module State
# =============================================================================

_last_request_ts: float = 0.0
_REQUEST_THROTTLE_SECONDS = 0.5
_REQUEST_MAX_ATTEMPTS = 3


# =============================================================================
# Utility Functions
# =============================================================================


def _ensure_dir(path: str) -> None:
    """Ensure directory exists for the given path."""
    directory = os.path.dirname(os.fspath(path)) or "."
    os.makedirs(directory, exist_ok=True)


def _is_stale(path: str, ttl: int) -> bool:
    """Return True if path is missing or older than ttl seconds."""
    try:
        mtime = os.path.getmtime(path)
    except FileNotFoundError:
        return True
    return (time.time() - mtime) > ttl


def _get_api_credentials() -> tuple:
    """Get Alpaca API credentials from environment."""
    api_key = os.environ.get("ALPACA_API_KEY", "")
    api_secret = os.environ.get("ALPACA_API_SECRET", "")
    return api_key, api_secret


# =============================================================================
# Main Functions
# =============================================================================


def fetch_from_alpaca(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    paper: bool = True,
    exchange_filter: Optional[str] = None,
    fractionable_only: bool = False,
    shortable_only: bool = False,
) -> List[Dict[str, Any]]:
    """
    Fetch tradable assets from Alpaca API.

    Args:
        api_key: Alpaca API key (or from env)
        api_secret: Alpaca API secret (or from env)
        paper: Use paper trading endpoint
        exchange_filter: Filter by exchange (NYSE, NASDAQ, AMEX)
        fractionable_only: Only include fractionable stocks
        shortable_only: Only include shortable stocks

    Returns:
        List of asset dictionaries

    Raises:
        ImportError: If alpaca-py not installed
        ValueError: If credentials not provided
    """
    # Get credentials
    if not api_key or not api_secret:
        api_key, api_secret = _get_api_credentials()

    if not api_key or not api_secret:
        raise ValueError(
            "Alpaca API credentials required. Set ALPACA_API_KEY and ALPACA_API_SECRET "
            "environment variables."
        )

    # Import Alpaca SDK
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetClass, AssetStatus
    except ImportError:
        raise ImportError("alpaca-py not installed. Install with: pip install alpaca-py")

    logger.info("Fetching stock universe from Alpaca...")

    client = TradingClient(
        api_key=api_key,
        secret_key=api_secret,
        paper=paper,
    )

    # Fetch all active US equities
    request = GetAssetsRequest(
        asset_class=AssetClass.US_EQUITY,
        status=AssetStatus.ACTIVE,
    )

    assets = client.get_all_assets(request)

    # Filter and convert to dict
    result = []
    for asset in assets:
        # Skip non-tradable
        if not asset.tradable:
            continue

        # Apply filters
        if exchange_filter:
            if str(asset.exchange).upper() != exchange_filter.upper():
                continue

        if fractionable_only and not asset.fractionable:
            continue

        if shortable_only and not asset.shortable:
            continue

        asset_dict = {
            "symbol": str(asset.symbol),
            "name": str(asset.name) if asset.name else "",
            "exchange": str(asset.exchange),
            "asset_class": "us_equity",
            "tradable": asset.tradable,
            "fractionable": getattr(asset, "fractionable", False),
            "marginable": getattr(asset, "marginable", False),
            "shortable": getattr(asset, "shortable", False),
            "easy_to_borrow": getattr(asset, "easy_to_borrow", None),
        }
        result.append(asset_dict)

    logger.info(f"Fetched {len(result)} tradable assets from Alpaca")
    return result


def run(
    out: str = DEFAULT_UNIVERSE_PATH,
    config: Optional[StockUniverseConfig] = None,
) -> List[str]:
    """
    Fetch stock symbols from Alpaca and store them.

    Args:
        out: Destination JSON file
        config: Configuration options

    Returns:
        Sorted list of symbols
    """
    config = config or StockUniverseConfig()

    try:
        assets = fetch_from_alpaca(
            paper=config.paper,
            exchange_filter=config.default_exchange,
            fractionable_only=config.fractionable_only,
            shortable_only=config.shortable_only,
        )
    except Exception as e:
        logger.error(f"Failed to fetch from Alpaca: {e}")
        # Return empty list on failure
        return []

    symbols = sorted([a["symbol"] for a in assets])

    # Save to file
    _ensure_dir(out)
    output_data = {
        "vendor": "alpaca",
        "asset_class": "us_equity",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "count": len(symbols),
        "symbols": symbols,
        "assets": assets,
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved {len(symbols)} symbols to {out}")
    return symbols


def get_symbols(
    ttl: int = DEFAULT_TTL_SECONDS,
    out: str = DEFAULT_UNIVERSE_PATH,
    force: bool = False,
    config: Optional[StockUniverseConfig] = None,
) -> List[str]:
    """
    Return cached stock symbols list, refreshing if needed.

    This is the main API for getting stock symbols with auto-refresh.

    Args:
        ttl: Cache TTL in seconds
        out: Cache file path
        force: Force refresh even if cache is fresh
        config: Configuration options

    Returns:
        List of stock symbols
    """
    if force or _is_stale(out, ttl):
        try:
            return run(out, config)
        except Exception as e:
            logger.warning(f"Failed to refresh stock universe: {e}")
            # Try to use existing cache
            if os.path.exists(out):
                with open(out, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("symbols", [])
            return []

    # Load from cache
    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("symbols", [])


def get_assets(
    ttl: int = DEFAULT_TTL_SECONDS,
    out: str = DEFAULT_UNIVERSE_PATH,
    force: bool = False,
    exchange: Optional[str] = None,
    config: Optional[StockUniverseConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Return cached asset details with optional filtering.

    Args:
        ttl: Cache TTL in seconds
        out: Cache file path
        force: Force refresh
        exchange: Filter by exchange
        config: Configuration options

    Returns:
        List of asset dictionaries
    """
    if force or _is_stale(out, ttl):
        try:
            run(out, config)
        except Exception as e:
            logger.warning(f"Failed to refresh stock universe: {e}")

    if not os.path.exists(out):
        return []

    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)
        assets = data.get("assets", [])

    # Apply exchange filter
    if exchange:
        assets = [a for a in assets if a.get("exchange", "").upper() == exchange.upper()]

    return assets


def get_popular_symbols(
    include_etfs: bool = True,
    highly_liquid_only: bool = False,
) -> List[str]:
    """
    Get curated list of popular/liquid stocks.

    This is a static list of well-known, liquid stocks suitable for:
    - Initial testing and development
    - Training models without full universe
    - High-frequency strategies needing low slippage

    Args:
        include_etfs: Include ETFs in the list
        highly_liquid_only: Only include highly liquid stocks

    Returns:
        Sorted list of popular symbols
    """
    if highly_liquid_only:
        symbols = HIGHLY_LIQUID_SYMBOLS
    else:
        symbols = POPULAR_SYMBOLS

    if not include_etfs:
        etfs = {
            "SPY",
            "QQQ",
            "IWM",
            "DIA",
            "VTI",
            "VOO",
            "VXX",
            "ARKK",
            "XLF",
            "XLE",
            "XLK",
            "XLV",
            "XLI",
            "XLC",
            "XLY",
            "XLP",
            "XLB",
            "XLU",
            "XLRE",
            "GLD",
            "IAU",
            "SGOL",
            "SLV",
            "TLT",
            "IEF",
            "SHY",
            "BND",
            "AGG",
        }
        symbols = symbols - etfs

    return sorted(symbols)


def get_sector_symbols(sector: str) -> List[str]:
    """
    Get symbols for a specific sector.

    Args:
        sector: Sector name (tech, finance, healthcare, consumer, energy, industrial)

    Returns:
        List of symbols in that sector
    """
    sectors = {
        "tech": [
            "AAPL",
            "MSFT",
            "GOOGL",
            "GOOG",
            "AMZN",
            "META",
            "NVDA",
            "TSLA",
            "AMD",
            "INTC",
            "CRM",
            "ORCL",
            "ADBE",
            "CSCO",
            "AVGO",
            "TXN",
            "QCOM",
            "ASML",
            "AMAT",
            "MU",
        ],
        "finance": [
            "JPM",
            "BAC",
            "WFC",
            "GS",
            "MS",
            "C",
            "BRK.B",
            "V",
            "MA",
            "AXP",
            "BLK",
            "SCHW",
            "USB",
            "PNC",
            "TFC",
        ],
        "healthcare": [
            "JNJ",
            "UNH",
            "PFE",
            "ABBV",
            "MRK",
            "LLY",
            "TMO",
            "ABT",
            "BMY",
            "AMGN",
            "GILD",
            "CVS",
            "CI",
            "HCA",
        ],
        "consumer": [
            "WMT",
            "COST",
            "HD",
            "MCD",
            "NKE",
            "SBUX",
            "TGT",
            "LOW",
            "DIS",
            "NFLX",
            "PG",
            "KO",
            "PEP",
            "PM",
            "MO",
        ],
        "energy": [
            "XOM",
            "CVX",
            "COP",
            "SLB",
            "EOG",
            "OXY",
            "MPC",
            "VLO",
            "PSX",
            "KMI",
        ],
        "industrial": [
            "CAT",
            "DE",
            "HON",
            "UNP",
            "UPS",
            "FDX",
            "BA",
            "LMT",
            "RTX",
            "GE",
        ],
    }

    return sectors.get(sector.lower(), [])


def get_etf_symbols(category: Optional[str] = None) -> List[str]:
    """
    Get ETF symbols by category.

    Args:
        category: ETF category (index, sector, commodity, bond)
                 If None, returns all ETFs

    Returns:
        List of ETF symbols
    """
    etf_categories = {
        "index": ["SPY", "QQQ", "IWM", "DIA", "VTI", "VOO"],
        "sector": ["XLF", "XLE", "XLK", "XLV", "XLI", "XLC", "XLY", "XLP", "XLB", "XLU", "XLRE"],
        "commodity": ["GLD", "IAU", "SGOL", "SLV"],
        "bond": ["TLT", "IEF", "SHY", "BND", "AGG"],
        "volatility": ["VXX"],
        "thematic": ["ARKK"],
    }

    if category:
        return etf_categories.get(category.lower(), [])

    # Return all ETFs
    all_etfs = []
    for etfs in etf_categories.values():
        all_etfs.extend(etfs)
    return sorted(set(all_etfs))


def refresh_universe(
    out: str = DEFAULT_UNIVERSE_PATH,
    config: Optional[StockUniverseConfig] = None,
) -> bool:
    """
    Force refresh the stock universe cache.

    Args:
        out: Output file path
        config: Configuration options

    Returns:
        True if refresh successful
    """
    try:
        symbols = run(out, config)
        return len(symbols) > 0
    except Exception as e:
        logger.error(f"Failed to refresh universe: {e}")
        return False


def get_universe_metadata(
    out: str = DEFAULT_UNIVERSE_PATH,
) -> Dict[str, Any]:
    """
    Get metadata about the cached universe.

    Args:
        out: Cache file path

    Returns:
        Metadata dict
    """
    if not os.path.exists(out):
        return {
            "exists": False,
            "count": 0,
        }

    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)

    mtime = datetime.fromtimestamp(os.path.getmtime(out))

    return {
        "exists": True,
        "count": data.get("count", len(data.get("symbols", []))),
        "vendor": data.get("vendor", "unknown"),
        "generated_at": data.get("generated_at"),
        "last_modified": mtime.isoformat(),
        "file_path": out,
    }


def filter_symbols_by_prefix(
    symbols: List[str],
    prefix: str,
) -> List[str]:
    """
    Filter symbols by prefix.

    Useful for filtering by ticker pattern (e.g., all symbols starting with 'A').

    Args:
        symbols: List of symbols
        prefix: Prefix to filter by

    Returns:
        Filtered list
    """
    return [s for s in symbols if s.startswith(prefix.upper())]


def is_symbol_tradable(
    symbol: str,
    out: str = DEFAULT_UNIVERSE_PATH,
) -> bool:
    """
    Check if a symbol is in the tradable universe.

    Args:
        symbol: Stock symbol
        out: Universe file path

    Returns:
        True if symbol is tradable
    """
    symbols = get_symbols(out=out)
    return symbol.upper() in symbols


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "get_symbols",
    "get_assets",
    "get_popular_symbols",
    "get_sector_symbols",
    "get_etf_symbols",
    "refresh_universe",
    "get_universe_metadata",
    "is_symbol_tradable",
    "fetch_from_alpaca",
    "run",
    "StockUniverseConfig",
    "POPULAR_SYMBOLS",
    "HIGHLY_LIQUID_SYMBOLS",
]
