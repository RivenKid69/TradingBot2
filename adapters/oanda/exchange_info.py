# -*- coding: utf-8 -*-
"""
adapters/oanda/exchange_info.py
OANDA exchange info adapter for forex currency pairs.

Status: Tested and operational (Phase 2 Complete)

Provides information about tradable currency pairs:
- Available instruments
- Pip sizes and precision
- Minimum/maximum trade sizes
- Margin requirements
- Trading status

OANDA v20 API Endpoints:
- GET /v3/accounts/{accountID}/instruments - List tradable instruments
- GET /v3/instruments/{instrument} - Instrument details

Currency Pair Classifications:
- Majors: G7 pairs (EUR/USD, USD/JPY, GBP/USD, etc.)
- Minors: Cross pairs without USD (EUR/GBP, EUR/CHF)
- Crosses: JPY crosses (EUR/JPY, GBP/JPY)
- Exotics: EM currencies (USD/TRY, USD/ZAR, USD/MXN)

Usage:
    adapter = OandaExchangeInfoAdapter(config={
        "api_key": "your_key",
        "account_id": "your_account",
    })

    # Get all tradable pairs
    pairs = adapter.get_tradable_symbols()

    # Get specific pair info
    info = adapter.get_symbol_info("EUR_USD")
    print(f"Pip: {info.tick_size}, Min: {info.min_qty}")

References:
    - OANDA Instruments API: https://developer.oanda.com/rest-live-v20/instrument-ep/
    - BIS Currency Classifications: https://www.bis.org/statistics/rpfx22.htm
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional, Set

from adapters.base import ExchangeInfoAdapter
from adapters.models import (
    ExchangeRule,
    ExchangeVendor,
    MarketType,
    SymbolInfo,
)

logger = logging.getLogger(__name__)


# =========================
# Currency Pair Classifications
# =========================

# Major pairs (G7 currencies, most liquid)
MAJOR_PAIRS: Set[str] = {
    "EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF",
    "AUD_USD", "USD_CAD", "NZD_USD",
}

# Minor pairs (no USD, G10 currencies)
MINOR_PAIRS: Set[str] = {
    "EUR_GBP", "EUR_CHF", "GBP_CHF", "EUR_AUD",
    "EUR_CAD", "EUR_NZD", "GBP_AUD", "GBP_CAD",
    "GBP_NZD", "AUD_NZD", "AUD_CHF", "AUD_CAD",
    "NZD_CHF", "NZD_CAD", "CHF_JPY",
}

# JPY crosses
CROSS_PAIRS: Set[str] = {
    "EUR_JPY", "GBP_JPY", "AUD_JPY", "NZD_JPY",
    "CAD_JPY",
}

# Exotic pairs (EM currencies)
EXOTIC_PAIRS: Set[str] = {
    "USD_TRY", "USD_ZAR", "USD_MXN", "USD_PLN",
    "USD_HUF", "USD_CZK", "USD_SGD", "USD_HKD",
    "USD_NOK", "USD_SEK", "USD_DKK", "EUR_TRY",
    "EUR_PLN", "EUR_HUF", "EUR_CZK", "EUR_NOK",
    "EUR_SEK", "EUR_DKK", "GBP_NOK", "GBP_SEK",
}

# All known forex pairs
ALL_FOREX_PAIRS: Set[str] = MAJOR_PAIRS | MINOR_PAIRS | CROSS_PAIRS | EXOTIC_PAIRS


# =========================
# Default Instrument Settings
# =========================

@dataclass
class ForexInstrumentDefaults:
    """Default settings for forex instruments."""
    pip_size: Decimal
    display_precision: int
    min_trade_size: Decimal  # Minimum units
    max_trade_size: Decimal  # Maximum units
    margin_rate: float       # Initial margin rate (e.g., 0.02 = 50:1 leverage)


# JPY pairs have different pip size
JPY_DEFAULTS = ForexInstrumentDefaults(
    pip_size=Decimal("0.01"),
    display_precision=3,
    min_trade_size=Decimal("1"),
    max_trade_size=Decimal("10000000"),
    margin_rate=0.02,  # 50:1
)

# Standard pairs
STANDARD_DEFAULTS = ForexInstrumentDefaults(
    pip_size=Decimal("0.0001"),
    display_precision=5,
    min_trade_size=Decimal("1"),
    max_trade_size=Decimal("10000000"),
    margin_rate=0.02,  # 50:1
)

# Exotic pairs (higher margin)
EXOTIC_DEFAULTS = ForexInstrumentDefaults(
    pip_size=Decimal("0.0001"),
    display_precision=5,
    min_trade_size=Decimal("1"),
    max_trade_size=Decimal("1000000"),
    margin_rate=0.10,  # 10:1
)


class OandaExchangeInfoAdapter(ExchangeInfoAdapter):
    """
    OANDA exchange info adapter for forex instruments.

    Provides instrument details including pip sizes, trade limits,
    and margin requirements.

    Configuration:
        api_key: OANDA API access token
        account_id: Trading account ID
        practice: Use practice environment (default: True)
        use_api: Fetch instruments from API (default: False, use static)

    Features:
    - Static instrument definitions for offline use
    - API-based instrument fetching when connected
    - Currency pair classification (major/minor/cross/exotic)
    - Pip size and precision lookup
    - Margin requirement estimation
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.OANDA,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Initialize OANDA exchange info adapter.

        Args:
            vendor: Exchange vendor (default: OANDA)
            config: Configuration dict
        """
        super().__init__(vendor, config)

        self._api_key = self._config.get("api_key") or os.getenv("OANDA_API_KEY")
        self._account_id = self._config.get("account_id") or os.getenv("OANDA_ACCOUNT_ID")
        self._practice = self._config.get("practice", True)
        self._use_api = self._config.get("use_api", False)

        # Cache for instrument info
        self._instruments_cache: Dict[str, ExchangeRule] = {}
        self._symbols_cache: Optional[List[str]] = None

        # Initialize with static data
        self._init_static_instruments()

    def _init_static_instruments(self) -> None:
        """Initialize instruments from static definitions."""
        for symbol in ALL_FOREX_PAIRS:
            self._instruments_cache[symbol] = self._create_exchange_rule(symbol)

    def _create_exchange_rule(self, symbol: str) -> ExchangeRule:
        """
        Create ExchangeRule for a forex pair.

        Args:
            symbol: Currency pair (e.g., "EUR_USD")

        Returns:
            ExchangeRule with trading parameters
        """
        # Get defaults based on pair type
        if "JPY" in symbol:
            defaults = JPY_DEFAULTS
        elif symbol in EXOTIC_PAIRS:
            defaults = EXOTIC_DEFAULTS
        else:
            defaults = STANDARD_DEFAULTS

        # Parse base/quote currencies
        parts = symbol.split("_")
        base_asset = parts[0] if len(parts) >= 2 else symbol[:3]
        quote_asset = parts[1] if len(parts) >= 2 else symbol[3:]

        return ExchangeRule(
            symbol=symbol,
            tick_size=defaults.pip_size,
            step_size=defaults.min_trade_size,
            min_notional=Decimal("1"),  # Forex typically has no min notional
            min_qty=defaults.min_trade_size,
            max_qty=defaults.max_trade_size,
            price_precision=defaults.display_precision,
            qty_precision=0,  # Forex uses whole units
            market_type=MarketType.FOREX,
            base_asset=base_asset,
            quote_asset=quote_asset,
            lot_size=1,
            is_tradable=True,
            is_marginable=True,
            is_shortable=True,  # Forex is always shortable
        )

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to OANDA format."""
        return symbol.upper().replace("/", "_").replace("-", "_")

    def get_tradable_symbols(self) -> List[str]:
        """
        Get list of all tradable currency pairs.

        Returns:
            List of symbol strings (e.g., ["EUR_USD", "GBP_USD", ...])
        """
        if self._symbols_cache is None:
            if self._use_api and self._api_key:
                self._fetch_instruments_from_api()
            else:
                self._symbols_cache = list(self._instruments_cache.keys())

        return self._symbols_cache or []

    def get_symbols(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Get list of tradable symbols with optional filters.

        Args:
            filters: Optional filters:
                - quote_asset: Filter by quote currency
                - base_asset: Filter by base currency
                - pair_type: Filter by type (major/minor/cross/exotic)
                - is_tradable: Only tradable symbols

        Returns:
            List of symbol strings
        """
        symbols = self.get_tradable_symbols()

        if not filters:
            return symbols

        result = []
        for symbol in symbols:
            rule = self._instruments_cache.get(symbol)
            if rule is None:
                continue

            # Apply filters
            if "quote_asset" in filters:
                if rule.quote_asset != filters["quote_asset"]:
                    continue

            if "base_asset" in filters:
                if rule.base_asset != filters["base_asset"]:
                    continue

            if "pair_type" in filters:
                if self.classify_pair(symbol) != filters["pair_type"]:
                    continue

            if "is_tradable" in filters:
                if rule.is_tradable != filters["is_tradable"]:
                    continue

            result.append(symbol)

        return result

    def get_exchange_rules(self, symbol: str) -> Optional[ExchangeRule]:
        """
        Get trading rules for a currency pair.

        This is the abstract method implementation required by ExchangeInfoAdapter.

        Args:
            symbol: Currency pair

        Returns:
            ExchangeRule or None if symbol not found
        """
        return self.get_exchange_rule(symbol)

    def refresh(self) -> bool:
        """
        Refresh cached exchange info from API.

        Returns:
            True if refresh successful
        """
        try:
            self._symbols_cache = None
            self._instruments_cache.clear()

            # Re-initialize with static data
            self._init_static_instruments()

            # If API is configured, fetch from API
            if self._use_api and self._api_key:
                self._fetch_instruments_from_api()

            return True
        except Exception as e:
            logger.error(f"Failed to refresh exchange info: {e}")
            return False

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """
        Get detailed information for a currency pair.

        Args:
            symbol: Currency pair (e.g., "EUR_USD", "EUR/USD")

        Returns:
            SymbolInfo or None if symbol not found
        """
        norm = self._normalize_symbol(symbol)

        rule = self._instruments_cache.get(norm)
        if rule is None:
            # Try to create on-the-fly if it looks like a valid pair
            if len(norm) == 7 and "_" in norm:
                rule = self._create_exchange_rule(norm)
                self._instruments_cache[norm] = rule
            else:
                return None

        return SymbolInfo(
            symbol=rule.symbol,
            vendor=self._vendor,
            market_type=MarketType.FOREX,
            exchange_rule=rule,
            fee_schedule=None,
            name=f"{rule.base_asset}/{rule.quote_asset}",
            status="active",
        )

    def get_exchange_rule(self, symbol: str) -> Optional[ExchangeRule]:
        """
        Get trading rules for a currency pair.

        Args:
            symbol: Currency pair

        Returns:
            ExchangeRule or None if symbol not found
        """
        norm = self._normalize_symbol(symbol)
        return self._instruments_cache.get(norm)

    def get_pip_size(self, symbol: str) -> Decimal:
        """
        Get pip size for a currency pair.

        Args:
            symbol: Currency pair

        Returns:
            Pip size (0.0001 for most, 0.01 for JPY pairs)
        """
        norm = self._normalize_symbol(symbol)
        rule = self._instruments_cache.get(norm)
        if rule:
            return rule.tick_size
        return Decimal("0.01") if "JPY" in norm else Decimal("0.0001")

    def get_margin_rate(self, symbol: str) -> float:
        """
        Get margin requirement rate for a currency pair.

        Args:
            symbol: Currency pair

        Returns:
            Margin rate (e.g., 0.02 = 50:1 leverage)
        """
        norm = self._normalize_symbol(symbol)

        if norm in EXOTIC_PAIRS:
            return 0.10  # 10:1 leverage for exotics
        elif norm in MAJOR_PAIRS:
            return 0.02  # 50:1 for majors
        else:
            return 0.033  # ~30:1 for minors/crosses

    def get_max_leverage(self, symbol: str) -> float:
        """
        Get maximum leverage for a currency pair.

        Args:
            symbol: Currency pair

        Returns:
            Maximum leverage (e.g., 50 for 50:1)
        """
        margin_rate = self.get_margin_rate(symbol)
        return 1.0 / margin_rate if margin_rate > 0 else 1.0

    def classify_pair(self, symbol: str) -> str:
        """
        Classify a currency pair type.

        Args:
            symbol: Currency pair

        Returns:
            Classification: "major", "minor", "cross", or "exotic"
        """
        norm = self._normalize_symbol(symbol)

        if norm in MAJOR_PAIRS:
            return "major"
        elif norm in MINOR_PAIRS:
            return "minor"
        elif norm in CROSS_PAIRS:
            return "cross"
        elif norm in EXOTIC_PAIRS:
            return "exotic"
        else:
            # Classify based on currencies
            parts = norm.split("_")
            if len(parts) != 2:
                return "unknown"

            base, quote = parts

            # G7 currencies
            g7 = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"}
            exotic_currencies = {"TRY", "ZAR", "MXN", "PLN", "HUF", "CZK", "THB", "SGD", "HKD"}

            if base in exotic_currencies or quote in exotic_currencies:
                return "exotic"
            elif "USD" in (base, quote):
                return "major"
            elif "JPY" in (base, quote):
                return "cross"
            else:
                return "minor"

    def is_tradable(self, symbol: str) -> bool:
        """
        Check if a currency pair is tradable.

        Args:
            symbol: Currency pair

        Returns:
            True if tradable
        """
        norm = self._normalize_symbol(symbol)
        rule = self._instruments_cache.get(norm)
        return rule is not None and rule.is_tradable

    def get_all_majors(self) -> List[str]:
        """Get all major currency pairs."""
        return sorted(list(MAJOR_PAIRS))

    def get_all_minors(self) -> List[str]:
        """Get all minor currency pairs."""
        return sorted(list(MINOR_PAIRS))

    def get_all_crosses(self) -> List[str]:
        """Get all JPY cross pairs."""
        return sorted(list(CROSS_PAIRS))

    def get_all_exotics(self) -> List[str]:
        """Get all exotic pairs."""
        return sorted(list(EXOTIC_PAIRS))

    def _fetch_instruments_from_api(self) -> None:
        """Fetch instruments from OANDA API."""
        if not self._api_key or not self._account_id:
            logger.warning("API credentials not provided, using static instruments")
            self._symbols_cache = list(self._instruments_cache.keys())
            return

        try:
            import requests

            # Determine URL based on environment
            if self._practice:
                base_url = "https://api-fxpractice.oanda.com"
            else:
                base_url = "https://api-fxtrade.oanda.com"

            url = f"{base_url}/v3/accounts/{self._account_id}/instruments"
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            instruments = data.get("instruments", [])
            self._symbols_cache = []

            for inst in instruments:
                symbol = inst.get("name", "")
                if not symbol:
                    continue

                # Parse instrument details
                pip_location = inst.get("pipLocation", -4)
                pip_size = Decimal("10") ** pip_location

                min_size = Decimal(str(inst.get("minimumTradeSize", "1")))
                max_size = Decimal(str(inst.get("maximumOrderUnits", "10000000")))
                margin_rate = float(inst.get("marginRate", "0.02"))

                parts = symbol.split("_")
                base_asset = parts[0] if len(parts) >= 2 else ""
                quote_asset = parts[1] if len(parts) >= 2 else ""

                rule = ExchangeRule(
                    symbol=symbol,
                    tick_size=pip_size,
                    step_size=min_size,
                    min_notional=Decimal("0"),
                    min_qty=min_size,
                    max_qty=max_size,
                    price_precision=abs(pip_location) + 1,
                    qty_precision=0,
                    market_type=MarketType.FOREX,
                    base_asset=base_asset,
                    quote_asset=quote_asset,
                    lot_size=1,
                    is_tradable=True,
                    is_marginable=True,
                    is_shortable=True,
                )

                self._instruments_cache[symbol] = rule
                self._symbols_cache.append(symbol)

            logger.info(f"Loaded {len(self._symbols_cache)} instruments from OANDA API")

        except Exception as e:
            logger.warning(f"Failed to fetch instruments from API: {e}")
            self._symbols_cache = list(self._instruments_cache.keys())

    def validate_order(
        self,
        symbol: str,
        qty: float,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Validate order parameters against exchange rules.

        Args:
            symbol: Currency pair
            qty: Order quantity
            price: Order price (optional)

        Returns:
            Dict with validation result and any errors
        """
        norm = self._normalize_symbol(symbol)
        rule = self._instruments_cache.get(norm)

        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        if rule is None:
            result["valid"] = False
            result["errors"].append(f"Unknown symbol: {symbol}")
            return result

        # Check quantity
        qty_dec = Decimal(str(qty))

        if qty_dec < rule.min_qty:
            result["valid"] = False
            result["errors"].append(
                f"Quantity {qty} below minimum {rule.min_qty}"
            )

        if rule.max_qty and qty_dec > rule.max_qty:
            result["valid"] = False
            result["errors"].append(
                f"Quantity {qty} exceeds maximum {rule.max_qty}"
            )

        # Check price precision if provided
        if price is not None:
            price_dec = Decimal(str(price))
            tick = rule.tick_size

            # Check if price is on tick
            remainder = price_dec % tick
            if remainder != 0:
                result["warnings"].append(
                    f"Price {price} not on tick size {tick}"
                )

        return result
