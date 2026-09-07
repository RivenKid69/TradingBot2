# -*- coding: utf-8 -*-
"""
adapters/binance/futures_exchange_info.py
Binance Futures exchange info adapter.

Provides futures-specific exchange information including:
- Contract specifications (tick size, lot size, multiplier)
- Leverage brackets (tiered margin)
- Position limits
- Trading rules and filters

References:
    - Binance Futures API: https://binance-docs.github.io/apidocs/futures/en/
    - Exchange Info: https://binance-docs.github.io/apidocs/futures/en/#exchange-information
    - Leverage Brackets: https://binance-docs.github.io/apidocs/futures/en/#notional-and-leverage-brackets
"""

from __future__ import annotations

import json
import logging
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from core_futures import (
    FuturesContractSpec,
    FuturesType,
    ContractType,
    Exchange,
    SettlementType,
    LeverageBracket,
)
from adapters.base import ExchangeInfoAdapter
from adapters.models import (
    ExchangeRule,
    ExchangeVendor,
    MarketType,
    SymbolInfo,
)

logger = logging.getLogger(__name__)


class BinanceFuturesExchangeInfoAdapter(ExchangeInfoAdapter):
    """
    Binance Futures exchange info adapter.

    Provides contract specifications, leverage brackets, and trading rules.

    Configuration:
        futures_url: API base URL (default: https://fapi.binance.com)
        filters_cache_path: Path to cached filters JSON
        timeout: Request timeout in seconds
        testnet: Use testnet URL
        auto_refresh: Auto-refresh on first access

    Example:
        >>> adapter = BinanceFuturesExchangeInfoAdapter()
        >>> spec = adapter.get_contract_spec("BTCUSDT")
        >>> print(f"Tick size: {spec.tick_size}, Max leverage: {spec.max_leverage}")
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.BINANCE,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor, config)

        # Cached data
        self._symbols_cache: Dict[str, SymbolInfo] = {}
        self._contract_specs: Dict[str, FuturesContractSpec] = {}
        self._leverage_brackets: Dict[str, List[LeverageBracket]] = {}
        self._exchange_info_raw: Optional[Dict[str, Any]] = None
        self._last_refresh_ts: int = 0

        # URLs
        self._futures_url = self._config.get("futures_url", "https://fapi.binance.com")
        if self._config.get("testnet", False):
            self._futures_url = "https://testnet.binancefuture.com"

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
                        info = self._parse_symbol_info(symbol, raw_filter)
                        if info:
                            self._symbols_cache[symbol] = info

                        spec = self._parse_contract_spec_from_cache(symbol, raw_filter)
                        if spec:
                            self._contract_specs[symbol] = spec

                self._exchange_info_raw = data
                logger.debug(f"Loaded {len(self._symbols_cache)} futures symbols from cache")
                return True

        except Exception as e:
            logger.warning(f"Failed to load futures exchange info cache: {e}")

        return False

    # ========================================================================
    # Standard ExchangeInfoAdapter methods
    # ========================================================================

    def get_symbols(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Get list of tradable futures symbols.

        Args:
            filters: Optional filters:
                - quote_asset: Filter by quote currency (e.g., "USDT")
                - base_asset: Filter by base currency
                - contract_type: Filter by contract type (e.g., "PERPETUAL")
                - is_tradable: Only tradable symbols (default: True)

        Returns:
            List of symbol strings
        """
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

            if filters.get("contract_type"):
                spec = self._contract_specs.get(symbol)
                if spec and spec.contract_type.value != filters["contract_type"]:
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
        if not self._symbols_cache:
            if self._config.get("auto_refresh", True):
                self.refresh()

        return self._symbols_cache.get(symbol.upper())

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
        Refresh exchange info from Binance Futures API.

        Returns:
            True if refresh successful
        """
        try:
            from binance_public import BinancePublicClient, PublicEndpoints

            endpoints = PublicEndpoints(
                spot_base=self._config.get("base_url", "https://api.binance.com"),
                futures_base=self._futures_url,
            )

            client = BinancePublicClient(
                endpoints=endpoints,
                timeout=int(self._config.get("timeout", 30)),
            )

            try:
                # Fetch futures exchange info
                exchange_info = client.get_exchange_info(use_futures=True)

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

                            spec = self._parse_contract_spec(symbol_data)
                            if spec:
                                self._contract_specs[symbol] = spec
                        except Exception as e:
                            logger.debug(f"Failed to parse symbol {symbol}: {e}")

                    self._exchange_info_raw = exchange_info
                    self._last_refresh_ts = int(time.time() * 1000)

                    logger.info(
                        f"Refreshed futures exchange info: {len(self._symbols_cache)} symbols"
                    )

                # Also fetch leverage brackets
                self._refresh_leverage_brackets(client)

                return True

            finally:
                client.close()

        except Exception as e:
            logger.error(f"Failed to refresh futures exchange info: {e}")
            return False

    def _refresh_leverage_brackets(self, client) -> None:
        """Fetch and cache leverage brackets for all symbols."""
        try:
            url = f"{self._futures_url}/fapi/v1/leverageBracket"

            data = client._session_get(
                url,
                params=None,
                budget="leverageBracket",
                tokens=1.0,
            )

            for item in data if isinstance(data, list) else []:
                if not isinstance(item, dict):
                    continue

                symbol = str(item.get("symbol", ""))
                brackets_data = item.get("brackets", [])

                brackets = []
                for b in brackets_data:
                    if not isinstance(b, dict):
                        continue

                    brackets.append(
                        LeverageBracket(
                            bracket=int(b.get("bracket", 0)),
                            notional_cap=Decimal(str(b.get("notionalCap", "0"))),
                            maint_margin_rate=Decimal(str(b.get("maintMarginRatio", "0.004"))),
                            max_leverage=int(b.get("initialLeverage", 125)),
                            cum_maintenance=Decimal(str(b.get("cum", "0"))),
                        )
                    )

                if brackets:
                    self._leverage_brackets[symbol] = brackets

            logger.debug(f"Loaded leverage brackets for {len(self._leverage_brackets)} symbols")

        except Exception as e:
            logger.warning(f"Failed to fetch leverage brackets: {e}")

    # ========================================================================
    # Futures-specific methods
    # ========================================================================

    def get_contract_spec(self, symbol: str) -> Optional[FuturesContractSpec]:
        """
        Get contract specification for a symbol.

        Provides unified contract spec with tick size, multiplier,
        margin requirements, and other contract details.

        Args:
            symbol: Futures symbol

        Returns:
            FuturesContractSpec or None
        """
        if not self._contract_specs:
            if self._config.get("auto_refresh", True):
                self.refresh()

        return self._contract_specs.get(symbol.upper())

    def get_leverage_brackets(self, symbol: str) -> List[LeverageBracket]:
        """
        Get leverage brackets for a symbol.

        Binance uses tiered margin where larger positions require
        more margin (lower max leverage).

        Args:
            symbol: Futures symbol

        Returns:
            List of LeverageBracket sorted by notional cap
        """
        if not self._leverage_brackets:
            if self._config.get("auto_refresh", True):
                self.refresh()

        brackets = self._leverage_brackets.get(symbol.upper(), [])
        return sorted(brackets, key=lambda b: b.notional_cap)

    def get_max_leverage(self, symbol: str, notional: Optional[Decimal] = None) -> int:
        """
        Get maximum leverage for a symbol at given notional.

        Args:
            symbol: Futures symbol
            notional: Position notional (uses first bracket if None)

        Returns:
            Maximum allowed leverage
        """
        brackets = self.get_leverage_brackets(symbol)
        if not brackets:
            # Default from contract spec
            spec = self.get_contract_spec(symbol)
            return spec.max_leverage if spec else 125

        if notional is None:
            return brackets[0].max_leverage

        # Find applicable bracket
        for bracket in brackets:
            if notional <= bracket.notional_cap:
                return bracket.max_leverage

        # Over max bracket, use last
        return brackets[-1].max_leverage

    def get_margin_requirement(
        self,
        symbol: str,
        notional: Decimal,
        leverage: int,
    ) -> Dict[str, Decimal]:
        """
        Calculate margin requirement for a position.

        Uses tiered bracket system for maintenance margin.

        Args:
            symbol: Futures symbol
            notional: Position notional value
            leverage: Requested leverage

        Returns:
            Dict with initial, maintenance, and available margin
        """
        brackets = self.get_leverage_brackets(symbol)

        # Calculate initial margin based on leverage
        initial_margin = notional / Decimal(leverage)

        # Calculate maintenance margin from brackets
        maint_margin = Decimal("0")
        for bracket in brackets:
            if notional <= bracket.notional_cap:
                maint_margin = notional * bracket.maint_margin_rate - bracket.cum_maintenance
                break

        # Fallback if no bracket found
        if maint_margin == 0 and brackets:
            last_bracket = brackets[-1]
            maint_margin = notional * last_bracket.maint_margin_rate - last_bracket.cum_maintenance

        return {
            "initial": initial_margin,
            "maintenance": max(maint_margin, Decimal("0")),
            "notional": notional,
            "leverage": Decimal(leverage),
        }

    def get_perpetual_symbols(self) -> List[str]:
        """Get all perpetual contract symbols."""
        return self.get_symbols({"contract_type": "PERPETUAL"})

    def get_quarterly_symbols(self) -> List[str]:
        """Get all quarterly (delivery) contract symbols."""
        result = []
        for symbol, spec in self._contract_specs.items():
            if spec.contract_type in (ContractType.CURRENT_QUARTER, ContractType.NEXT_QUARTER):
                result.append(symbol)
        return sorted(result)

    def get_all_contract_specs(self) -> Dict[str, FuturesContractSpec]:
        """Get all contract specifications."""
        if not self._contract_specs:
            if self._config.get("auto_refresh", True):
                self.refresh()

        return dict(self._contract_specs)

    # ========================================================================
    # Parse methods
    # ========================================================================

    def _parse_symbol_info(
        self,
        symbol: str,
        raw_filter: Dict[str, Any],
    ) -> Optional[SymbolInfo]:
        """Parse cached filter data into SymbolInfo."""
        tick_size = self._extract_decimal(raw_filter, "tick_size", "0.01")
        step_size = self._extract_decimal(raw_filter, "step_size", "0.001")
        min_notional = self._extract_decimal(raw_filter, "min_notional", "5")
        min_qty = self._extract_decimal(raw_filter, "min_qty", "0.001")
        max_qty = self._extract_decimal(raw_filter, "max_qty", None)
        price_precision = int(raw_filter.get("price_precision", 2))
        qty_precision = int(raw_filter.get("qty_precision", 3))

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
            market_type=MarketType.CRYPTO_FUTURES,
            base_asset=base_asset,
            quote_asset=quote_asset,
            is_tradable=is_tradable,
            raw_filters=raw_filter,
        )

        return SymbolInfo(
            symbol=symbol,
            vendor=self._vendor,
            market_type=MarketType.CRYPTO_FUTURES,
            exchange_rule=exchange_rule,
            status="active" if is_tradable else "inactive",
            raw_data=raw_filter,
        )

    def _parse_exchange_info_symbol(
        self,
        symbol_data: Dict[str, Any],
    ) -> Optional[SymbolInfo]:
        """Parse Binance futures exchangeInfo symbol into SymbolInfo."""
        symbol = symbol_data.get("symbol", "")
        if not symbol:
            return None

        # Extract base/quote
        base_asset = str(symbol_data.get("baseAsset", ""))
        quote_asset = str(symbol_data.get("quoteAsset", ""))

        # Parse filters
        filters = symbol_data.get("filters", [])
        raw_filters: Dict[str, Any] = {}

        tick_size = Decimal("0.01")
        step_size = Decimal("0.001")
        min_notional = Decimal("5")
        min_qty = Decimal("0.001")
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

            elif filter_type == "MIN_NOTIONAL":
                min_notional = self._safe_decimal(f.get("notional"), min_notional)

        # Precision
        price_precision = int(symbol_data.get("pricePrecision", 2))
        qty_precision = int(symbol_data.get("quantityPrecision", 3))

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
            market_type=MarketType.CRYPTO_FUTURES,
            base_asset=base_asset,
            quote_asset=quote_asset,
            is_tradable=is_tradable,
            raw_filters=raw_filters,
        )

        return SymbolInfo(
            symbol=symbol,
            vendor=self._vendor,
            market_type=MarketType.CRYPTO_FUTURES,
            exchange_rule=exchange_rule,
            status="active" if is_tradable else "inactive",
            raw_data=symbol_data,
        )

    def _parse_contract_spec(self, symbol_data: Dict[str, Any]) -> Optional[FuturesContractSpec]:
        """Parse Binance futures exchangeInfo into FuturesContractSpec."""
        symbol = symbol_data.get("symbol", "")
        if not symbol:
            return None

        base_asset = str(symbol_data.get("baseAsset", ""))
        quote_asset = str(symbol_data.get("quoteAsset", ""))
        margin_asset = str(symbol_data.get("marginAsset", quote_asset))

        # Determine contract type
        contract_type_str = str(symbol_data.get("contractType", "PERPETUAL")).upper()
        if contract_type_str == "PERPETUAL":
            contract_type = ContractType.PERPETUAL
            futures_type = FuturesType.CRYPTO_PERPETUAL
            settlement_type = SettlementType.FUNDING
        elif contract_type_str == "CURRENT_QUARTER":
            contract_type = ContractType.CURRENT_QUARTER
            futures_type = FuturesType.CRYPTO_QUARTERLY
            settlement_type = SettlementType.CASH
        elif contract_type_str == "NEXT_QUARTER":
            contract_type = ContractType.NEXT_QUARTER
            futures_type = FuturesType.CRYPTO_QUARTERLY
            settlement_type = SettlementType.CASH
        else:
            contract_type = ContractType.PERPETUAL
            futures_type = FuturesType.CRYPTO_PERPETUAL
            settlement_type = SettlementType.FUNDING

        # Parse filters for tick/lot size
        filters = symbol_data.get("filters", [])
        tick_size = Decimal("0.01")
        lot_size = Decimal("0.001")
        min_qty = Decimal("0.001")
        max_qty = Decimal("1000")

        for f in filters:
            if not isinstance(f, dict):
                continue
            filter_type = f.get("filterType", "")

            if filter_type == "PRICE_FILTER":
                tick_size = self._safe_decimal(f.get("tickSize"), tick_size)

            elif filter_type == "LOT_SIZE":
                lot_size = self._safe_decimal(f.get("stepSize"), lot_size)
                min_qty = self._safe_decimal(f.get("minQty"), min_qty)
                max_qty = self._safe_decimal(f.get("maxQty"), max_qty)

        # Delivery date for quarterly
        delivery_date = symbol_data.get("deliveryDate")
        if delivery_date and isinstance(delivery_date, int):
            # Convert ms to YYYYMMDD
            import datetime

            dt = datetime.datetime.fromtimestamp(delivery_date / 1000, tz=datetime.timezone.utc)
            delivery_date = dt.strftime("%Y%m%d")

        return FuturesContractSpec(
            symbol=symbol,
            futures_type=futures_type,
            contract_type=contract_type,
            exchange=Exchange.BINANCE,
            base_asset=base_asset,
            quote_asset=quote_asset,
            margin_asset=margin_asset,
            contract_size=Decimal("1"),
            multiplier=Decimal("1"),
            tick_size=tick_size,
            tick_value=tick_size,  # For crypto, tick_value = tick_size
            min_qty=min_qty,
            max_qty=max_qty,
            lot_size=lot_size,
            max_leverage=125,  # Default, updated from leverage brackets
            initial_margin_pct=Decimal("0.8"),  # 125x -> 0.8%
            maint_margin_pct=Decimal("0.4"),
            settlement_type=settlement_type,
            delivery_date=delivery_date if contract_type != ContractType.PERPETUAL else None,
            liquidation_fee_pct=Decimal("0.5"),
            maker_fee_bps=Decimal("2.0"),
            taker_fee_bps=Decimal("4.0"),
            trading_hours="24/7",
        )

    def _parse_contract_spec_from_cache(
        self,
        symbol: str,
        raw_filter: Dict[str, Any],
    ) -> Optional[FuturesContractSpec]:
        """Parse contract spec from cached filter data."""
        base_asset = str(raw_filter.get("base_asset", ""))
        quote_asset = str(raw_filter.get("quote_asset", "USDT"))
        margin_asset = str(raw_filter.get("margin_asset", quote_asset))

        contract_type_str = str(raw_filter.get("contract_type", "PERPETUAL")).upper()
        if contract_type_str == "PERPETUAL":
            contract_type = ContractType.PERPETUAL
            futures_type = FuturesType.CRYPTO_PERPETUAL
            settlement_type = SettlementType.FUNDING
        else:
            contract_type = ContractType.CURRENT_QUARTER
            futures_type = FuturesType.CRYPTO_QUARTERLY
            settlement_type = SettlementType.CASH

        tick_size = self._extract_decimal(raw_filter, "tick_size", "0.01")
        lot_size = self._extract_decimal(raw_filter, "step_size", "0.001")
        min_qty = self._extract_decimal(raw_filter, "min_qty", "0.001")
        max_qty = self._extract_decimal(raw_filter, "max_qty", "1000")
        max_leverage = int(raw_filter.get("max_leverage", 125))

        return FuturesContractSpec(
            symbol=symbol,
            futures_type=futures_type,
            contract_type=contract_type,
            exchange=Exchange.BINANCE,
            base_asset=base_asset,
            quote_asset=quote_asset,
            margin_asset=margin_asset,
            contract_size=Decimal("1"),
            multiplier=Decimal("1"),
            tick_size=tick_size,
            tick_value=tick_size,
            min_qty=min_qty,
            max_qty=max_qty,
            lot_size=lot_size,
            max_leverage=max_leverage,
            initial_margin_pct=Decimal("100") / Decimal(max_leverage),
            maint_margin_pct=Decimal("50") / Decimal(max_leverage),
            settlement_type=settlement_type,
            delivery_date=raw_filter.get("delivery_date"),
            liquidation_fee_pct=Decimal("0.5"),
            maker_fee_bps=Decimal("2.0"),
            taker_fee_bps=Decimal("4.0"),
            trading_hours="24/7",
        )

    # ========================================================================
    # Helper methods
    # ========================================================================

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
            filters = {}
            for symbol, info in self._symbols_cache.items():
                rule = info.exchange_rule
                spec = self._contract_specs.get(symbol)

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
                    "contract_type": spec.contract_type.value if spec else "PERPETUAL",
                    "max_leverage": spec.max_leverage if spec else 125,
                    "margin_asset": spec.margin_asset if spec else rule.quote_asset,
                }

            data = {
                "metadata": {
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "source": "binance_futures",
                    "vendor": self._vendor.value,
                    "market_type": "CRYPTO_FUTURES",
                },
                "filters": filters,
                "leverage_brackets": {
                    symbol: [b.to_dict() for b in brackets]
                    for symbol, brackets in self._leverage_brackets.items()
                },
            }

            with open(save_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(filters)} futures symbols to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save futures exchange info cache: {e}")
            return False

    @property
    def market_type(self) -> MarketType:
        """Return the market type this adapter serves."""
        return MarketType.CRYPTO_FUTURES
