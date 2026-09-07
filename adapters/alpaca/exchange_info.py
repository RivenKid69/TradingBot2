# -*- coding: utf-8 -*-
"""
adapters/alpaca/exchange_info.py
Alpaca exchange info adapter for US equities.

Provides stock information, trading rules, and metadata.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional

from adapters.base import ExchangeInfoAdapter
from adapters.models import (
    ExchangeRule,
    ExchangeVendor,
    FeeSchedule,
    FeeStructure,
    MarketType,
    SymbolInfo,
)

logger = logging.getLogger(__name__)


class AlpacaExchangeInfoAdapter(ExchangeInfoAdapter):
    """
    Alpaca exchange info adapter for US equities.

    Provides asset/symbol information from Alpaca Markets.

    Configuration:
        api_key: Alpaca API key (required)
        api_secret: Alpaca API secret (required)
        paper: Use paper trading endpoint (default: False)
        auto_refresh: Auto-refresh on first access (default: True)
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.ALPACA,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor, config)

        # Cached data
        self._assets_cache: Dict[str, SymbolInfo] = {}
        self._client = None

    def _get_client(self):
        """Lazy initialization of Alpaca client."""
        if self._client is None:
            try:
                from alpaca.trading.client import TradingClient
            except ImportError:
                raise ImportError("Alpaca SDK not installed. Install with: pip install alpaca-py")

            api_key = self._config.get("api_key")
            api_secret = self._config.get("api_secret")
            paper = self._config.get("paper", False)

            if not api_key or not api_secret:
                raise ValueError("Alpaca API key and secret are required")

            self._client = TradingClient(
                api_key=api_key,
                secret_key=api_secret,
                paper=paper,
            )

        return self._client

    def get_symbols(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Get list of tradable stock symbols.

        Args:
            filters: Optional filters:
                - asset_class: "us_equity" (default)
                - exchange: Filter by exchange ("NYSE", "NASDAQ", "AMEX")
                - is_tradable: Only tradable symbols (default: True)
                - is_fractionable: Only fractional-eligible symbols

        Returns:
            List of symbol strings
        """
        # Ensure we have data
        if not self._assets_cache:
            if self._config.get("auto_refresh", True):
                self.refresh()

        if filters is None:
            filters = {}

        result = []
        for symbol, info in self._assets_cache.items():
            # Apply filters
            if filters.get("exchange"):
                exchange = info.raw_data.get("exchange", "")
                if exchange != filters["exchange"]:
                    continue

            if filters.get("is_tradable", True):
                if not info.is_tradable:
                    continue

            if filters.get("is_fractionable"):
                if not info.is_fractionable:
                    continue

            result.append(symbol)

        return sorted(result)

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """
        Get detailed symbol information.

        Args:
            symbol: Stock symbol (e.g., "AAPL")

        Returns:
            SymbolInfo or None if not found
        """
        # Ensure we have data
        if not self._assets_cache:
            if self._config.get("auto_refresh", True):
                self.refresh()

        symbol_upper = symbol.upper()
        return self._assets_cache.get(symbol_upper)

    def get_exchange_rules(self, symbol: str) -> Optional[ExchangeRule]:
        """
        Get trading rules for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            ExchangeRule or None
        """
        info = self.get_symbol_info(symbol)
        return info.exchange_rule if info else None

    def refresh(self) -> bool:
        """
        Refresh asset list from Alpaca API.

        Returns:
            True if refresh successful
        """
        try:
            client = self._get_client()

            from alpaca.trading.requests import GetAssetsRequest
            from alpaca.trading.enums import AssetClass, AssetStatus

            # Request tradable US equities
            request = GetAssetsRequest(
                asset_class=AssetClass.US_EQUITY,
                status=AssetStatus.ACTIVE,
            )

            assets = client.get_all_assets(request)

            for asset in assets:
                try:
                    info = self._parse_asset(asset)
                    if info:
                        self._assets_cache[info.symbol] = info
                except Exception as e:
                    logger.debug(f"Failed to parse asset {asset.symbol}: {e}")

            logger.info(f"Refreshed Alpaca assets: {len(self._assets_cache)} symbols")
            return True

        except Exception as e:
            logger.error(f"Failed to refresh Alpaca assets: {e}")
            return False

    def _parse_asset(self, asset: Any) -> Optional[SymbolInfo]:
        """Parse Alpaca asset into SymbolInfo."""
        if not asset.tradable:
            return None

        symbol = str(asset.symbol)

        # US equities have standard rules
        # Tick size: $0.01 for most stocks
        # Step size: 1 share (or fractional if enabled)
        # No min notional (Alpaca allows any size)

        tick_size = Decimal("0.01")
        step_size = Decimal("0.0001") if asset.fractionable else Decimal("1")
        min_qty = Decimal("0.0001") if asset.fractionable else Decimal("1")

        exchange_rule = ExchangeRule(
            symbol=symbol,
            tick_size=tick_size,
            step_size=step_size,
            min_notional=Decimal("0"),  # No minimum
            min_qty=min_qty,
            max_qty=None,
            price_precision=2,
            qty_precision=4 if asset.fractionable else 0,
            market_type=MarketType.EQUITY,
            base_asset=symbol,
            quote_asset="USD",
            lot_size=1,
            is_tradable=asset.tradable,
            is_marginable=asset.marginable,
            is_shortable=asset.shortable,
            raw_filters={},
        )

        return SymbolInfo(
            symbol=symbol,
            vendor=self._vendor,
            market_type=MarketType.EQUITY,
            exchange_rule=exchange_rule,
            fee_schedule=FeeSchedule(
                structure=FeeStructure.FLAT,
                maker_rate=0.0,
                taker_rate=0.0,
                currency="USD",
            ),
            name=str(asset.name) if asset.name else "",
            is_etf=getattr(asset, "is_etf", False) if hasattr(asset, "is_etf") else False,
            is_fractionable=asset.fractionable,
            status="active" if asset.tradable else "inactive",
            raw_data={
                "id": str(asset.id),
                "exchange": str(asset.exchange),
                "asset_class": str(asset.asset_class),
                "easy_to_borrow": getattr(asset, "easy_to_borrow", None),
                "maintenance_margin_requirement": getattr(
                    asset, "maintenance_margin_requirement", None
                ),
            },
        )

    def get_asset_by_id(self, asset_id: str) -> Optional[SymbolInfo]:
        """
        Get asset by Alpaca asset ID.

        Args:
            asset_id: Alpaca asset UUID

        Returns:
            SymbolInfo or None
        """
        try:
            client = self._get_client()
            asset = client.get_asset(asset_id)
            return self._parse_asset(asset)
        except Exception as e:
            logger.warning(f"Failed to get asset {asset_id}: {e}")
            return None

    def search_symbols(self, query: str, limit: int = 10) -> List[str]:
        """
        Search for symbols matching query.

        Args:
            query: Search string
            limit: Maximum results

        Returns:
            List of matching symbols
        """
        if not self._assets_cache:
            self.refresh()

        query_upper = query.upper()
        matches = []

        for symbol, info in self._assets_cache.items():
            if symbol.startswith(query_upper):
                matches.append((0, symbol))  # Prefix match = priority 0
            elif query_upper in symbol:
                matches.append((1, symbol))  # Contains = priority 1
            elif info.name and query_upper in info.name.upper():
                matches.append((2, symbol))  # Name match = priority 2

        # Sort by priority, then alphabetically
        matches.sort(key=lambda x: (x[0], x[1]))

        return [m[1] for m in matches[:limit]]
