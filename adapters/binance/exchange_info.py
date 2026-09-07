# -*- coding: utf-8 -*-
"""
adapters/binance/exchange_info.py
Binance exchange info adapter.

Provides symbol information, trading rules, and exchange metadata.
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from pathlib import Path
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


class BinanceExchangeInfoAdapter(ExchangeInfoAdapter):
    """
    Binance exchange info adapter.

    Provides symbol information, trading rules, and filters.

    Configuration:
        base_url: API base URL (default: https://api.binance.com)
        futures_url: Futures API URL (default: https://fapi.binance.com)
        filters_cache_path: Path to cached filters JSON
        use_futures: Whether to use futures endpoints (default: False)
        auto_refresh: Auto-refresh on first access (default: True)
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.BINANCE,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor, config)

        # Cached data
        self._symbols_cache: Dict[str, SymbolInfo] = {}
        self._exchange_info_raw: Optional[Dict[str, Any]] = None
        self._last_refresh_ts: int = 0

        # Load from cache if available
        cache_path = self._config.get("filters_cache_path")
        if cache_path:
            self._load_from_cache(cache_path)

    def _load_from_cache(self, path: str) -> bool:
        """Load exchange info from cache file."""
        try:
            cache_file = Path(path)
            if not cache_file.exists():
                return False

            with open(cache_file, "r") as f:
                data = json.load(f)

            if isinstance(data, dict):
                filters = data.get("filters", data)
                for symbol, raw_filter in filters.items():
                    if isinstance(raw_filter, dict):
                        self._symbols_cache[symbol] = self._parse_symbol_info(symbol, raw_filter)

                self._exchange_info_raw = data
                logger.debug(f"Loaded {len(self._symbols_cache)} symbols from cache")
                return True

        except Exception as e:
            logger.warning(f"Failed to load exchange info cache: {e}")

        return False

    def get_symbols(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Get list of tradable symbols.

        Args:
            filters: Optional filters:
                - quote_asset: Filter by quote currency (e.g., "USDT")
                - base_asset: Filter by base currency
                - is_tradable: Only tradable symbols (default: True)
                - min_volume: Minimum 24h volume

        Returns:
            List of symbol strings
        """
        # Ensure we have data
        if not self._symbols_cache:
            if self._config.get("auto_refresh", True):
                self.refresh()

        if filters is None:
            filters = {}

        result = []
        for symbol, info in self._symbols_cache.items():
            # Apply filters
            if filters.get("quote_asset"):
                if info.exchange_rule.quote_asset != filters["quote_asset"]:
                    continue

            if filters.get("base_asset"):
                if info.exchange_rule.base_asset != filters["base_asset"]:
                    continue

            if filters.get("is_tradable", True):
                if not info.is_tradable:
                    continue

            result.append(symbol)

        return sorted(result)

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """
        Get detailed symbol information.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")

        Returns:
            SymbolInfo or None if not found
        """
        # Ensure we have data
        if not self._symbols_cache:
            if self._config.get("auto_refresh", True):
                self.refresh()

        symbol_upper = symbol.upper()
        return self._symbols_cache.get(symbol_upper)

    def get_exchange_rules(self, symbol: str) -> Optional[ExchangeRule]:
        """
        Get trading rules for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            ExchangeRule or None if not found
        """
        info = self.get_symbol_info(symbol)
        return info.exchange_rule if info else None

    def refresh(self) -> bool:
        """
        Refresh exchange info from Binance API.

        Returns:
            True if refresh successful
        """
        try:
            from binance_public import BinancePublicClient, PublicEndpoints

            endpoints = PublicEndpoints(
                spot_base=self._config.get("base_url", "https://api.binance.com"),
                futures_base=self._config.get("futures_url", "https://fapi.binance.com"),
            )

            client = BinancePublicClient(
                endpoints=endpoints,
                timeout=int(self._config.get("timeout", 30)),
            )

            try:
                # Fetch exchange info
                exchange_info = client.get_exchange_info(
                    use_futures=self._config.get("use_futures", False)
                )

                if isinstance(exchange_info, dict):
                    symbols_data = exchange_info.get("symbols", [])

                    for symbol_data in symbols_data:
                        if not isinstance(symbol_data, dict):
                            continue

                        symbol = symbol_data.get("symbol", "")
                        if not symbol:
                            continue

                        try:
                            info = self._parse_exchange_info_symbol(symbol_data)
                            if info:
                                self._symbols_cache[symbol] = info
                        except Exception as e:
                            logger.debug(f"Failed to parse symbol {symbol}: {e}")

                    self._exchange_info_raw = exchange_info
                    import time

                    self._last_refresh_ts = int(time.time() * 1000)

                    logger.info(f"Refreshed exchange info: {len(self._symbols_cache)} symbols")
                    return True

            finally:
                client.close()

        except Exception as e:
            logger.error(f"Failed to refresh exchange info: {e}")
            return False

        return False

    def _parse_symbol_info(
        self,
        symbol: str,
        raw_filter: Dict[str, Any],
    ) -> SymbolInfo:
        """Parse cached filter data into SymbolInfo."""
        # Extract fields with defaults
        tick_size = self._extract_decimal(raw_filter, "tick_size", "0.00000001")
        step_size = self._extract_decimal(raw_filter, "step_size", "0.00000001")
        min_notional = self._extract_decimal(raw_filter, "min_notional", "0")
        min_qty = self._extract_decimal(raw_filter, "min_qty", "0")
        max_qty = self._extract_decimal(raw_filter, "max_qty", None)
        price_precision = int(raw_filter.get("price_precision", 8))
        qty_precision = int(raw_filter.get("qty_precision", 8))

        base_asset = str(raw_filter.get("base_asset", ""))
        quote_asset = str(raw_filter.get("quote_asset", ""))

        is_tradable = raw_filter.get("status", "TRADING") == "TRADING"

        exchange_rule = ExchangeRule(
            symbol=symbol,
            tick_size=tick_size,
            step_size=step_size,
            min_notional=min_notional,
            min_qty=min_qty,
            max_qty=max_qty,
            price_precision=price_precision,
            qty_precision=qty_precision,
            market_type=MarketType.CRYPTO_SPOT,
            base_asset=base_asset,
            quote_asset=quote_asset,
            is_tradable=is_tradable,
            raw_filters=raw_filter,
        )

        return SymbolInfo(
            symbol=symbol,
            vendor=self._vendor,
            market_type=MarketType.CRYPTO_SPOT,
            exchange_rule=exchange_rule,
            status="active" if is_tradable else "inactive",
            raw_data=raw_filter,
        )

    def _parse_exchange_info_symbol(
        self,
        symbol_data: Dict[str, Any],
    ) -> Optional[SymbolInfo]:
        """Parse Binance exchangeInfo symbol into SymbolInfo."""
        symbol = symbol_data.get("symbol", "")
        if not symbol:
            return None

        # Extract base/quote
        base_asset = str(symbol_data.get("baseAsset", ""))
        quote_asset = str(symbol_data.get("quoteAsset", ""))

        # Parse filters
        filters = symbol_data.get("filters", [])
        raw_filters: Dict[str, Any] = {}

        tick_size = Decimal("0.00000001")
        step_size = Decimal("0.00000001")
        min_notional = Decimal("0")
        min_qty = Decimal("0")
        max_qty: Optional[Decimal] = None

        for f in filters:
            if not isinstance(f, dict):
                continue

            filter_type = f.get("filterType", "")
            raw_filters[filter_type] = f

            if filter_type == "PRICE_FILTER":
                tick_size = self._safe_decimal(f.get("tickSize"), tick_size)

            elif filter_type == "LOT_SIZE":
                step_size = self._safe_decimal(f.get("stepSize"), step_size)
                min_qty = self._safe_decimal(f.get("minQty"), min_qty)
                max_qty = self._safe_decimal(f.get("maxQty"), None)

            elif filter_type in ("MIN_NOTIONAL", "NOTIONAL"):
                min_notional = self._safe_decimal(
                    f.get("minNotional") or f.get("notional"), min_notional
                )

        # Precision
        price_precision = int(symbol_data.get("quotePrecision", 8))
        qty_precision = int(symbol_data.get("baseAssetPrecision", 8))

        # Status
        status = symbol_data.get("status", "TRADING")
        is_tradable = status == "TRADING"

        exchange_rule = ExchangeRule(
            symbol=symbol,
            tick_size=tick_size,
            step_size=step_size,
            min_notional=min_notional,
            min_qty=min_qty,
            max_qty=max_qty,
            price_precision=price_precision,
            qty_precision=qty_precision,
            market_type=MarketType.CRYPTO_SPOT,
            base_asset=base_asset,
            quote_asset=quote_asset,
            is_tradable=is_tradable,
            raw_filters=raw_filters,
        )

        return SymbolInfo(
            symbol=symbol,
            vendor=self._vendor,
            market_type=MarketType.CRYPTO_SPOT,
            exchange_rule=exchange_rule,
            status="active" if is_tradable else "inactive",
            raw_data=symbol_data,
        )

    @staticmethod
    def _extract_decimal(
        data: Dict[str, Any],
        key: str,
        default: Optional[str],
    ) -> Decimal:
        """Extract Decimal from dict with fallback."""
        value = data.get(key)
        if value is None:
            return Decimal(default) if default else Decimal("0")
        try:
            return Decimal(str(value))
        except Exception:
            return Decimal(default) if default else Decimal("0")

    @staticmethod
    def _safe_decimal(value: Any, default: Optional[Decimal]) -> Optional[Decimal]:
        """Safely convert to Decimal."""
        if value is None:
            return default
        try:
            return Decimal(str(value))
        except Exception:
            return default

    def get_exchange_filters_file(self) -> Optional[str]:
        """Get path to filters cache file."""
        return self._config.get("filters_cache_path")

    def save_to_cache(self, path: Optional[str] = None) -> bool:
        """
        Save current exchange info to cache file.

        Args:
            path: Cache file path (optional, uses config default)

        Returns:
            True if save successful
        """
        save_path = path or self._config.get("filters_cache_path")
        if not save_path:
            return False

        try:
            # Build filters dict
            filters = {}
            for symbol, info in self._symbols_cache.items():
                rule = info.exchange_rule
                filters[symbol] = {
                    "tick_size": str(rule.tick_size),
                    "step_size": str(rule.step_size),
                    "min_notional": str(rule.min_notional),
                    "min_qty": str(rule.min_qty),
                    "max_qty": str(rule.max_qty) if rule.max_qty else None,
                    "price_precision": rule.price_precision,
                    "qty_precision": rule.qty_precision,
                    "base_asset": rule.base_asset,
                    "quote_asset": rule.quote_asset,
                    "status": "TRADING" if rule.is_tradable else "INACTIVE",
                }

            import time

            data = {
                "metadata": {
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "source": "binance",
                    "vendor": self._vendor.value,
                },
                "filters": filters,
            }

            with open(save_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(filters)} symbols to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save exchange info cache: {e}")
            return False
