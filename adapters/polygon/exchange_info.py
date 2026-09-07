# -*- coding: utf-8 -*-
"""
adapters/polygon/exchange_info.py
Polygon.io exchange info adapter implementation.

Provides symbol metadata and trading rules from Polygon.io API:
- Ticker details (name, sector, industry)
- Exchange rules (tick size, lot size)
- Symbol search and filtering

API Reference:
    https://polygon.io/docs/stocks/get_v3_reference_tickers__ticker_
    https://polygon.io/docs/stocks/get_v3_reference_tickers
"""

from __future__ import annotations

import logging
import os
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..base import ExchangeInfoAdapter
from ..models import (
    ExchangeRule,
    ExchangeVendor,
    FeeSchedule,
    FeeStructure,
    MarketType,
    SymbolInfo,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DEFAULT EXCHANGE RULES FOR US EQUITIES
# =============================================================================

DEFAULT_EQUITY_RULE = ExchangeRule(
    symbol="",
    tick_size=Decimal("0.01"),  # Standard US equity tick size
    step_size=Decimal("1"),  # Whole shares (fractional supported by broker)
    min_notional=Decimal("1"),  # $1 minimum
    min_qty=Decimal("1"),  # 1 share minimum
    max_qty=None,  # No max
    price_precision=2,
    qty_precision=4,  # Fractional shares
    market_type=MarketType.EQUITY,
    lot_size=1,
    is_tradable=True,
    is_marginable=True,
    is_shortable=True,
)


# =============================================================================
# POLYGON EXCHANGE INFO ADAPTER
# =============================================================================


class PolygonExchangeInfoAdapter(ExchangeInfoAdapter):
    """
    Exchange info adapter for Polygon.io.

    Provides:
    - Ticker metadata (name, sector, industry)
    - Trading rules (tick size, lot size)
    - Symbol listing and search

    Configuration:
        api_key: Polygon.io API key
        cache_ttl: Symbol cache TTL in seconds (default: 3600)

    Example:
        adapter = PolygonExchangeInfoAdapter(config={"api_key": "..."})

        # Get symbol info
        info = adapter.get_symbol_info("AAPL")
        print(info.name, info.sector)

        # List symbols
        symbols = adapter.get_symbols(filters={"market_type": "EQUITY"})
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.POLYGON,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor=vendor, config=config)

        self._api_key = self._config.get("api_key") or os.environ.get("POLYGON_API_KEY", "")
        self._cache_ttl = float(self._config.get("cache_ttl", 3600))

        # Cache
        self._symbol_cache: Dict[str, SymbolInfo] = {}
        self._symbols_list_cache: Optional[List[str]] = None

        # REST client (lazy)
        self._rest_client: Optional[Any] = None

    def _get_rest_client(self) -> Any:
        """Lazy initialization of REST client."""
        if self._rest_client is None:
            if not self._api_key:
                raise ValueError("Polygon API key required")
            try:
                from polygon import RESTClient

                self._rest_client = RESTClient(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "polygon-api-client not installed. Run: pip install polygon-api-client"
                )
        return self._rest_client

    # -------------------------------------------------------------------------
    # ExchangeInfoAdapter Interface
    # -------------------------------------------------------------------------

    def get_symbols(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Get list of tradable symbols.

        Args:
            filters: Optional filters:
                - market_type: "stocks", "crypto", "fx", "options"
                - active: Only active symbols (default: True)
                - search: Search query
                - limit: Maximum results (default: 1000)

        Returns:
            List of symbol strings
        """
        filters = filters or {}

        # Use cached list if available
        if self._symbols_list_cache is not None:
            return self._apply_filters(self._symbols_list_cache, filters)

        try:
            client = self._get_rest_client()

            market = filters.get("market_type", "stocks")
            active = filters.get("active", True)
            search = filters.get("search")
            limit = filters.get("limit", 1000)

            # Fetch tickers
            tickers = client.list_tickers(
                market=market,
                active=active,
                search=search,
                limit=limit,
            )

            symbols = [t.ticker for t in tickers if hasattr(t, "ticker")]

            # Cache results
            self._symbols_list_cache = symbols

            return symbols

        except Exception as e:
            logger.error(f"Failed to get symbols: {e}")
            return []

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """
        Get detailed symbol information.

        Args:
            symbol: Stock ticker (e.g., "AAPL")

        Returns:
            SymbolInfo or None if not found
        """
        symbol = symbol.upper()

        # Check cache
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]

        try:
            client = self._get_rest_client()
            details = client.get_ticker_details(symbol)

            if details is None:
                return None

            # Build exchange rule
            exchange_rule = ExchangeRule(
                symbol=symbol,
                tick_size=Decimal("0.01"),
                step_size=Decimal("1"),
                min_notional=Decimal("1"),
                min_qty=Decimal("1"),
                max_qty=None,
                price_precision=2,
                qty_precision=4,
                market_type=MarketType.EQUITY,
                base_asset=symbol,
                quote_asset="USD",
                lot_size=1,
                is_tradable=getattr(details, "active", True),
                is_marginable=True,
                is_shortable=True,
                raw_filters={},
            )

            # Build fee schedule (Polygon is data-only, no fees)
            # Actual trading fees depend on broker
            fee_schedule = FeeSchedule(
                structure=FeeStructure.FLAT,
                maker_rate=0.0,
                taker_rate=0.0,
                flat_fee=0.0,
                currency="USD",
            )

            # Build symbol info
            info = SymbolInfo(
                symbol=symbol,
                vendor=ExchangeVendor.POLYGON,
                market_type=MarketType.EQUITY,
                exchange_rule=exchange_rule,
                fee_schedule=fee_schedule,
                name=getattr(details, "name", ""),
                sector=getattr(details, "sic_description", ""),
                industry=getattr(details, "sic_description", ""),
                is_etf=getattr(details, "type", "") == "ETF",
                is_fractionable=True,  # Broker-dependent
                status="active" if getattr(details, "active", True) else "inactive",
                listed_date=str(getattr(details, "list_date", "")),
                delisted_date=(
                    str(getattr(details, "delisted_utc", ""))
                    if hasattr(details, "delisted_utc")
                    else None
                ),
                raw_data=details.__dict__ if hasattr(details, "__dict__") else {},
            )

            # Cache
            self._symbol_cache[symbol] = info

            return info

        except Exception as e:
            logger.error(f"Failed to get symbol info for {symbol}: {e}")
            return None

    def get_exchange_rules(self, symbol: str) -> Optional[ExchangeRule]:
        """Get trading rules for a symbol."""
        info = self.get_symbol_info(symbol)
        return info.exchange_rule if info else None

    def refresh(self) -> bool:
        """Refresh cached exchange info."""
        self._symbol_cache.clear()
        self._symbols_list_cache = None
        logger.info("Polygon exchange info cache cleared")
        return True

    def get_symbols_info(
        self,
        symbols: list[str],
    ) -> Dict[str, Optional[SymbolInfo]]:
        """Get info for multiple symbols."""
        return {s: self.get_symbol_info(s) for s in symbols}

    # -------------------------------------------------------------------------
    # Additional Methods
    # -------------------------------------------------------------------------

    def search_symbols(
        self,
        query: str,
        limit: int = 20,
    ) -> List[SymbolInfo]:
        """
        Search for symbols by name or ticker.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching SymbolInfo
        """
        try:
            client = self._get_rest_client()
            tickers = client.list_tickers(
                search=query,
                active=True,
                limit=limit,
            )

            results: List[SymbolInfo] = []
            for t in tickers:
                if hasattr(t, "ticker"):
                    info = self.get_symbol_info(t.ticker)
                    if info:
                        results.append(info)

            return results

        except Exception as e:
            logger.error(f"Symbol search failed: {e}")
            return []

    def get_popular_symbols(
        self,
        limit: int = 100,
    ) -> List[str]:
        """
        Get list of popular/high-volume symbols.

        Note: This is a static list. For production, consider
        sorting by volume from market data.
        """
        popular = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            "BRK.B",
            "UNH",
            "JNJ",
            "XOM",
            "JPM",
            "V",
            "PG",
            "MA",
            "HD",
            "CVX",
            "MRK",
            "ABBV",
            "LLY",
            "AVGO",
            "PEP",
            "KO",
            "COST",
            "TMO",
            "WMT",
            "MCD",
            "CSCO",
            "ACN",
            "ABT",
            "DHR",
            "VZ",
            "ADBE",
            "CRM",
            "CMCSA",
            "NKE",
            "INTC",
            "TXN",
            "NEE",
            "PM",
            "QCOM",
            "UNP",
            "HON",
            "LOW",
            "IBM",
            "AMAT",
            "GE",
            "CAT",
            "BA",
            "AMD",
            "PYPL",
            # Popular ETFs
            "SPY",
            "QQQ",
            "IWM",
            "DIA",
            "VTI",
            "VOO",
            "EEM",
            "XLF",
            "XLE",
            "XLK",
            "XLV",
            "XLI",
            "GLD",
            "SLV",
            "TLT",
            "HYG",
            "LQD",
            "VNQ",
            "ARKK",
            "SOXL",
            "TQQQ",
        ]

        return popular[:limit]

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _apply_filters(
        self,
        symbols: List[str],
        filters: Dict[str, Any],
    ) -> List[str]:
        """Apply filters to symbol list."""
        result = symbols

        search = filters.get("search")
        if search:
            search_lower = search.lower()
            result = [s for s in result if search_lower in s.lower()]

        limit = filters.get("limit")
        if limit and limit > 0:
            result = result[:limit]

        return result
