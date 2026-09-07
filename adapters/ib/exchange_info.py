# -*- coding: utf-8 -*-
"""
adapters/ib/exchange_info.py
Interactive Brokers exchange info adapter for CME Group futures.

Provides contract specifications, trading rules, and exchange metadata
for CME, CBOT, NYMEX, and COMEX futures contracts.

Features:
- Contract details (multiplier, tick size, expiry)
- Trading rules and price limits
- Contract rollover information
- Exchange status and trading hours

Reference:
- IB Contract Info: https://interactivebrokers.github.io/tws-api/contracts.html
- CME Contract Specs: https://www.cmegroup.com/trading/products/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional, Sequence

from adapters.base import ExchangeInfoAdapter
from adapters.models import (
    ExchangeRule,
    ExchangeVendor,
    MarketType,
    SymbolInfo,
)

# Import core_futures models
from core_futures import (
    FuturesContractSpec,
    FuturesType,
    ContractType,
    SettlementType,
    Exchange,
)

logger = logging.getLogger(__name__)

# Try to import ib_insync
try:
    from ib_insync import IB, Future, ContFuture

    IB_INSYNC_AVAILABLE = True
except ImportError:
    IB_INSYNC_AVAILABLE = False
    IB = None
    Future = None
    ContFuture = None


# =========================
# Contract Specifications
# =========================

# HARDCODED CONTRACT SPECS - FALLBACK DATA
# =========================================
# Tech Debt: data-ib-hardcoded-specs (docs/reports/TECH_DEBT_REGISTRY.md)
#
# Purpose: Fallback contract specifications when IB connection unavailable
# Source: CME Group public specifications (https://www.cmegroup.com/trading/products/)
# Last Updated: 2025-12-21
# Version: 1.0.0
#
# IMPORTANT:
# - These values are APPROXIMATE and may become stale
# - Margin requirements change frequently based on market conditions
# - Always prefer live IB data when available
# - For production, verify against live broker specs
#
# Refresh procedure:
# 1. Check CME Group website for current specs
# 2. Update values below
# 3. Update "Last Updated" date above
# 4. Run tests: pytest tests/test_ib_adapters.py -v
#
CONTRACT_SPECS_VERSION = "1.0.0"
CONTRACT_SPECS_UPDATED = "2025-12-21"
CONTRACT_SPECS: Dict[str, Dict[str, Any]] = {
    # E-mini S&P 500
    "ES": {
        "exchange": "CME",
        "futures_type": FuturesType.INDEX_FUTURES,
        "multiplier": Decimal("50"),
        "tick_size": Decimal("0.25"),
        "tick_value": Decimal("12.50"),  # 50 * 0.25
        "currency": "USD",
        "description": "E-mini S&P 500 Futures",
        "trading_hours": "Sun-Fri: 6:00pm - 5:00pm ET (with daily break)",
        "initial_margin": Decimal("12650"),  # Approximate, varies
        "maintenance_margin": Decimal("11500"),
    },
    # E-mini NASDAQ 100
    "NQ": {
        "exchange": "CME",
        "futures_type": FuturesType.INDEX_FUTURES,
        "multiplier": Decimal("20"),
        "tick_size": Decimal("0.25"),
        "tick_value": Decimal("5.00"),
        "currency": "USD",
        "description": "E-mini NASDAQ 100 Futures",
        "trading_hours": "Sun-Fri: 6:00pm - 5:00pm ET (with daily break)",
        "initial_margin": Decimal("17600"),
        "maintenance_margin": Decimal("16000"),
    },
    # E-mini Dow Jones
    "YM": {
        "exchange": "CBOT",
        "futures_type": FuturesType.INDEX_FUTURES,
        "multiplier": Decimal("5"),
        "tick_size": Decimal("1"),
        "tick_value": Decimal("5.00"),
        "currency": "USD",
        "description": "E-mini Dow Jones Futures",
        "trading_hours": "Sun-Fri: 6:00pm - 5:00pm ET (with daily break)",
        "initial_margin": Decimal("9900"),
        "maintenance_margin": Decimal("9000"),
    },
    # Gold Futures
    "GC": {
        "exchange": "COMEX",
        "futures_type": FuturesType.COMMODITY_FUTURES,
        "multiplier": Decimal("100"),  # 100 troy ounces
        "tick_size": Decimal("0.10"),
        "tick_value": Decimal("10.00"),
        "currency": "USD",
        "description": "Gold Futures (100 oz)",
        "trading_hours": "Sun-Fri: 6:00pm - 5:00pm ET (with daily break)",
        "initial_margin": Decimal("10230"),
        "maintenance_margin": Decimal("9300"),
    },
    # Silver Futures
    "SI": {
        "exchange": "COMEX",
        "futures_type": FuturesType.COMMODITY_FUTURES,
        "multiplier": Decimal("5000"),  # 5000 troy ounces
        "tick_size": Decimal("0.005"),
        "tick_value": Decimal("25.00"),
        "currency": "USD",
        "description": "Silver Futures (5000 oz)",
        "trading_hours": "Sun-Fri: 6:00pm - 5:00pm ET (with daily break)",
        "initial_margin": Decimal("14850"),
        "maintenance_margin": Decimal("13500"),
    },
    # Crude Oil Futures
    "CL": {
        "exchange": "NYMEX",
        "futures_type": FuturesType.COMMODITY_FUTURES,
        "multiplier": Decimal("1000"),  # 1000 barrels
        "tick_size": Decimal("0.01"),
        "tick_value": Decimal("10.00"),
        "currency": "USD",
        "description": "WTI Crude Oil Futures (1000 bbl)",
        "trading_hours": "Sun-Fri: 6:00pm - 5:00pm ET (with daily break)",
        "initial_margin": Decimal("7040"),
        "maintenance_margin": Decimal("6400"),
    },
    # Natural Gas Futures
    "NG": {
        "exchange": "NYMEX",
        "futures_type": FuturesType.COMMODITY_FUTURES,
        "multiplier": Decimal("10000"),  # 10000 MMBtu
        "tick_size": Decimal("0.001"),
        "tick_value": Decimal("10.00"),
        "currency": "USD",
        "description": "Natural Gas Futures (10000 MMBtu)",
        "trading_hours": "Sun-Fri: 6:00pm - 5:00pm ET (with daily break)",
        "initial_margin": Decimal("3410"),
        "maintenance_margin": Decimal("3100"),
    },
    # Euro FX Futures
    "6E": {
        "exchange": "CME",
        "futures_type": FuturesType.CURRENCY_FUTURES,
        "multiplier": Decimal("125000"),  # 125000 EUR
        "tick_size": Decimal("0.00005"),
        "tick_value": Decimal("6.25"),
        "currency": "USD",
        "description": "Euro FX Futures (125000 EUR)",
        "trading_hours": "Sun-Fri: 6:00pm - 5:00pm ET (with daily break)",
        "initial_margin": Decimal("2530"),
        "maintenance_margin": Decimal("2300"),
    },
    # Japanese Yen Futures
    "6J": {
        "exchange": "CME",
        "futures_type": FuturesType.CURRENCY_FUTURES,
        "multiplier": Decimal("12500000"),  # 12.5M JPY
        "tick_size": Decimal("0.0000005"),
        "tick_value": Decimal("6.25"),
        "currency": "USD",
        "description": "Japanese Yen Futures (12.5M JPY)",
        "trading_hours": "Sun-Fri: 6:00pm - 5:00pm ET (with daily break)",
        "initial_margin": Decimal("3410"),
        "maintenance_margin": Decimal("3100"),
    },
    # British Pound Futures
    "6B": {
        "exchange": "CME",
        "futures_type": FuturesType.CURRENCY_FUTURES,
        "multiplier": Decimal("62500"),  # 62500 GBP
        "tick_size": Decimal("0.0001"),
        "tick_value": Decimal("6.25"),
        "currency": "USD",
        "description": "British Pound Futures (62500 GBP)",
        "trading_hours": "Sun-Fri: 6:00pm - 5:00pm ET (with daily break)",
        "initial_margin": Decimal("2640"),
        "maintenance_margin": Decimal("2400"),
    },
    # 10-Year Treasury Note
    "ZN": {
        "exchange": "CBOT",
        "futures_type": FuturesType.BOND_FUTURES,
        "multiplier": Decimal("1000"),  # $100,000 face value / 100
        "tick_size": Decimal("0.015625"),  # 1/64 of a point
        "tick_value": Decimal("15.625"),
        "currency": "USD",
        "description": "10-Year Treasury Note Futures",
        "trading_hours": "Sun-Fri: 6:00pm - 5:00pm ET (with daily break)",
        "initial_margin": Decimal("2200"),
        "maintenance_margin": Decimal("2000"),
    },
    # 30-Year Treasury Bond
    "ZB": {
        "exchange": "CBOT",
        "futures_type": FuturesType.BOND_FUTURES,
        "multiplier": Decimal("1000"),
        "tick_size": Decimal("0.03125"),  # 1/32 of a point
        "tick_value": Decimal("31.25"),
        "currency": "USD",
        "description": "30-Year Treasury Bond Futures",
        "trading_hours": "Sun-Fri: 6:00pm - 5:00pm ET (with daily break)",
        "initial_margin": Decimal("4070"),
        "maintenance_margin": Decimal("3700"),
    },
    # Micro E-mini S&P 500
    "MES": {
        "exchange": "CME",
        "futures_type": FuturesType.INDEX_FUTURES,
        "multiplier": Decimal("5"),  # 1/10 of ES
        "tick_size": Decimal("0.25"),
        "tick_value": Decimal("1.25"),
        "currency": "USD",
        "description": "Micro E-mini S&P 500 Futures",
        "trading_hours": "Sun-Fri: 6:00pm - 5:00pm ET (with daily break)",
        "initial_margin": Decimal("1265"),
        "maintenance_margin": Decimal("1150"),
    },
    # Micro E-mini NASDAQ 100
    "MNQ": {
        "exchange": "CME",
        "futures_type": FuturesType.INDEX_FUTURES,
        "multiplier": Decimal("2"),  # 1/10 of NQ
        "tick_size": Decimal("0.25"),
        "tick_value": Decimal("0.50"),
        "currency": "USD",
        "description": "Micro E-mini NASDAQ 100 Futures",
        "trading_hours": "Sun-Fri: 6:00pm - 5:00pm ET (with daily break)",
        "initial_margin": Decimal("1760"),
        "maintenance_margin": Decimal("1600"),
    },
    # Micro Gold
    "MGC": {
        "exchange": "COMEX",
        "futures_type": FuturesType.COMMODITY_FUTURES,
        "multiplier": Decimal("10"),  # 10 troy ounces (1/10 of GC)
        "tick_size": Decimal("0.10"),
        "tick_value": Decimal("1.00"),
        "currency": "USD",
        "description": "Micro Gold Futures (10 oz)",
        "trading_hours": "Sun-Fri: 6:00pm - 5:00pm ET (with daily break)",
        "initial_margin": Decimal("1023"),
        "maintenance_margin": Decimal("930"),
    },
    # Micro Crude Oil
    "MCL": {
        "exchange": "NYMEX",
        "futures_type": FuturesType.COMMODITY_FUTURES,
        "multiplier": Decimal("100"),  # 100 barrels (1/10 of CL)
        "tick_size": Decimal("0.01"),
        "tick_value": Decimal("1.00"),
        "currency": "USD",
        "description": "Micro Crude Oil Futures (100 bbl)",
        "trading_hours": "Sun-Fri: 6:00pm - 5:00pm ET (with daily break)",
        "initial_margin": Decimal("704"),
        "maintenance_margin": Decimal("640"),
    },
    # Australian Dollar Futures
    "6A": {
        "exchange": "CME",
        "futures_type": FuturesType.CURRENCY_FUTURES,
        "multiplier": Decimal("100000"),  # 100000 AUD
        "tick_size": Decimal("0.0001"),
        "tick_value": Decimal("10.00"),
        "currency": "USD",
        "description": "Australian Dollar Futures (100000 AUD)",
        "trading_hours": "Sun-Fri: 6:00pm - 5:00pm ET (with daily break)",
        "initial_margin": Decimal("1760"),
        "maintenance_margin": Decimal("1600"),
    },
    # Canadian Dollar Futures
    "6C": {
        "exchange": "CME",
        "futures_type": FuturesType.CURRENCY_FUTURES,
        "multiplier": Decimal("100000"),  # 100000 CAD
        "tick_size": Decimal("0.00005"),
        "tick_value": Decimal("5.00"),
        "currency": "USD",
        "description": "Canadian Dollar Futures (100000 CAD)",
        "trading_hours": "Sun-Fri: 6:00pm - 5:00pm ET (with daily break)",
        "initial_margin": Decimal("1430"),
        "maintenance_margin": Decimal("1300"),
    },
    # Swiss Franc Futures
    "6S": {
        "exchange": "CME",
        "futures_type": FuturesType.CURRENCY_FUTURES,
        "multiplier": Decimal("125000"),  # 125000 CHF
        "tick_size": Decimal("0.0001"),
        "tick_value": Decimal("12.50"),
        "currency": "USD",
        "description": "Swiss Franc Futures (125000 CHF)",
        "trading_hours": "Sun-Fri: 6:00pm - 5:00pm ET (with daily break)",
        "initial_margin": Decimal("3520"),
        "maintenance_margin": Decimal("3200"),
    },
    # 2-Year Treasury Note
    "ZT": {
        "exchange": "CBOT",
        "futures_type": FuturesType.BOND_FUTURES,
        "multiplier": Decimal("2000"),  # $200,000 face value / 100
        "tick_size": Decimal("0.0078125"),  # 1/128 of a point
        "tick_value": Decimal("15.625"),
        "currency": "USD",
        "description": "2-Year Treasury Note Futures",
        "trading_hours": "Sun-Fri: 6:00pm - 5:00pm ET (with daily break)",
        "initial_margin": Decimal("660"),
        "maintenance_margin": Decimal("600"),
    },
    # 5-Year Treasury Note
    "ZF": {
        "exchange": "CBOT",
        "futures_type": FuturesType.BOND_FUTURES,
        "multiplier": Decimal("1000"),  # $100,000 face value / 100
        "tick_size": Decimal("0.0078125"),  # 1/128 of a point
        "tick_value": Decimal("7.8125"),
        "currency": "USD",
        "description": "5-Year Treasury Note Futures",
        "trading_hours": "Sun-Fri: 6:00pm - 5:00pm ET (with daily break)",
        "initial_margin": Decimal("1100"),
        "maintenance_margin": Decimal("1000"),
    },
}


# =========================
# IB Exchange Info Adapter
# =========================


class IBExchangeInfoAdapter(ExchangeInfoAdapter):
    """
    Interactive Brokers exchange info adapter.

    Provides contract specifications and trading rules for CME Group futures.

    Configuration:
        host: TWS/Gateway host (default: 127.0.0.1)
        port: TWS port (7497 paper, 7496 live)
        client_id: Unique client ID
        default_exchange: Default exchange for symbol lookup

    Usage:
        adapter = IBExchangeInfoAdapter(
            vendor=ExchangeVendor.IB,
            config={"port": 7497}
        )
        adapter.connect()
        spec = adapter.get_contract_spec("ES")
        symbols = adapter.get_symbols(filters={"exchange": "CME"})
        adapter.disconnect()
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.IB,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor, config)
        self._ib: Optional[IB] = None
        self._host = self._config.get("host", "127.0.0.1")
        self._port = self._config.get("port", 7497)
        self._client_id = self._config.get("client_id", 3)
        self._default_exchange = self._config.get("default_exchange", "CME")

        # Cache for contract details
        self._contract_cache: Dict[str, FuturesContractSpec] = {}

    def _do_connect(self) -> None:
        """Connect to TWS/Gateway."""
        if not IB_INSYNC_AVAILABLE:
            raise ImportError(
                "ib_insync is required for IB adapters. " "Install with: pip install ib_insync"
            )

        self._ib = IB()
        self._ib.connect(
            self._host,
            self._port,
            clientId=self._client_id,
            readonly=True,
            timeout=self._config.get("timeout", 10.0),
        )
        logger.info(f"Connected to IB TWS for exchange info at {self._host}:{self._port}")

    def _do_disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
            logger.info("Disconnected from IB TWS")
        self._ib = None

    def get_contract_spec(
        self,
        symbol: str,
        fetch_from_ib: bool = True,
    ) -> Optional[FuturesContractSpec]:
        """
        Get futures contract specification.

        Args:
            symbol: Futures symbol (ES, NQ, GC, etc.)
            fetch_from_ib: If True, fetch from IB. If False, use hardcoded specs.

        Returns:
            FuturesContractSpec or None if not found
        """
        symbol = symbol.upper()

        # Check cache first
        if symbol in self._contract_cache:
            return self._contract_cache[symbol]

        # Try to fetch from IB
        if fetch_from_ib and self._ib and self._ib.isConnected():
            try:
                spec = self._fetch_contract_spec_from_ib(symbol)
                if spec:
                    self._contract_cache[symbol] = spec
                    return spec
            except Exception as e:
                logger.warning(f"Failed to fetch contract spec from IB: {e}")

        # Fall back to hardcoded specs
        if symbol in CONTRACT_SPECS:
            spec = self._create_spec_from_hardcoded(symbol)
            self._contract_cache[symbol] = spec
            return spec

        return None

    def _fetch_contract_spec_from_ib(self, symbol: str) -> Optional[FuturesContractSpec]:
        """Fetch contract specification from IB."""
        if not self._ib:
            return None

        # Get contract details from IB
        details_map = CONTRACT_SPECS.get(symbol, {})
        exchange = details_map.get("exchange", self._default_exchange)

        contract = Future(symbol, exchange=exchange)
        details_list = self._ib.reqContractDetails(contract)

        if not details_list:
            return None

        # Use first matching contract
        details = details_list[0]
        c = details.contract

        # Determine futures type
        futures_type = FuturesType.INDEX_FUTURES
        if symbol in ["GC", "SI", "HG", "CL", "NG"]:
            futures_type = FuturesType.COMMODITY_FUTURES
        elif symbol in ["6E", "6J", "6B", "6A", "6C"]:
            futures_type = FuturesType.CURRENCY_FUTURES
        elif symbol in ["ZN", "ZB", "ZT", "ZF"]:
            futures_type = FuturesType.BOND_FUTURES

        return FuturesContractSpec(
            symbol=symbol,
            futures_type=futures_type,
            contract_type=ContractType.CURRENT_QUARTER,
            exchange=Exchange(c.exchange),
            base_asset=c.symbol,
            quote_asset="USD",
            margin_asset="USD",
            settlement_type=(
                SettlementType.CASH
                if symbol in ["ES", "NQ", "YM", "RTY"]
                else SettlementType.PHYSICAL
            ),
            multiplier=Decimal(str(c.multiplier)) if c.multiplier else Decimal("1"),
            tick_size=Decimal(str(details.minTick)),
            min_qty=Decimal("1"),
            max_qty=Decimal("10000"),
            expiry_date=(
                c.lastTradeDateOrContractMonth[:10] if c.lastTradeDateOrContractMonth else None
            ),
        )

    def _create_spec_from_hardcoded(self, symbol: str) -> FuturesContractSpec:
        """Create contract spec from hardcoded data."""
        data = CONTRACT_SPECS[symbol]

        return FuturesContractSpec(
            symbol=symbol,
            futures_type=data["futures_type"],
            contract_type=ContractType.CURRENT_QUARTER,
            exchange=Exchange(data["exchange"]),
            base_asset=symbol,
            quote_asset="USD",
            margin_asset="USD",
            settlement_type=(
                SettlementType.CASH
                if symbol in ["ES", "NQ", "YM", "RTY"]
                else SettlementType.PHYSICAL
            ),
            multiplier=data["multiplier"],
            tick_size=data["tick_size"],
            min_qty=Decimal("1"),
            max_qty=Decimal("10000"),
        )

    # =========================
    # Base Class Implementation
    # =========================

    def get_symbols(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Get list of tradable futures symbols.

        Args:
            filters: Optional filters:
                - exchange: Filter by exchange (CME, CBOT, NYMEX, COMEX)
                - futures_type: Filter by type (INDEX, COMMODITY, CURRENCY)

        Returns:
            List of symbol strings
        """
        symbols = list(CONTRACT_SPECS.keys())

        if not filters:
            return symbols

        # Apply filters
        if "exchange" in filters:
            exchange = filters["exchange"].upper()
            symbols = [s for s in symbols if CONTRACT_SPECS[s].get("exchange") == exchange]

        if "futures_type" in filters:
            ft = filters["futures_type"]
            if isinstance(ft, str):
                ft = FuturesType(ft.upper())
            symbols = [s for s in symbols if CONTRACT_SPECS[s].get("futures_type") == ft]

        return symbols

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """
        Get detailed symbol information.

        Args:
            symbol: Futures symbol

        Returns:
            SymbolInfo or None if symbol not found
        """
        symbol = symbol.upper()
        spec = self.get_contract_spec(symbol, fetch_from_ib=False)

        if not spec:
            return None

        data = CONTRACT_SPECS.get(symbol, {})

        return SymbolInfo(
            symbol=symbol,
            base_asset=symbol,
            quote_asset="USD",
            status="TRADING",
            is_tradable=True,
            market_type=self._futures_type_to_market_type(spec.futures_type),
            tick_size=float(spec.tick_size),
            min_qty=float(spec.min_qty),
            max_qty=float(spec.max_qty),
            min_notional=float(data.get("initial_margin", 0)),
            description=data.get("description", ""),
        )

    def get_exchange_rules(self, symbol: str) -> Optional[ExchangeRule]:
        """
        Get trading rules for a symbol.

        Args:
            symbol: Futures symbol

        Returns:
            ExchangeRule or None if symbol not found
        """
        symbol = symbol.upper()

        if symbol not in CONTRACT_SPECS:
            return None

        data = CONTRACT_SPECS[symbol]

        return ExchangeRule(
            symbol=symbol,
            tick_size=float(data.get("tick_size", 0.01)),
            min_qty=1.0,
            max_qty=10000.0,
            step_size=1.0,
            min_notional=float(data.get("initial_margin", 0)),
            max_notional=None,
            price_filter_min=None,
            price_filter_max=None,
            max_orders=None,
        )

    def refresh(self) -> bool:
        """
        Refresh cached exchange info.

        Returns:
            True if refresh successful
        """
        self._contract_cache.clear()
        return True

    def get_contract_expirations(
        self,
        symbol: str,
        num_contracts: int = 4,
    ) -> List[str]:
        """
        Get upcoming contract expiration dates.

        Args:
            symbol: Futures symbol
            num_contracts: Number of contracts to return

        Returns:
            List of expiration dates in YYYYMMDD format
        """
        if not self._ib or not self._ib.isConnected():
            return []

        symbol = symbol.upper()
        exchange = CONTRACT_SPECS.get(symbol, {}).get("exchange", self._default_exchange)

        contract = Future(symbol, exchange=exchange)
        details_list = self._ib.reqContractDetails(contract)

        expirations = []
        for details in details_list:
            exp = details.contract.lastTradeDateOrContractMonth
            if exp and len(exp) >= 8:
                expirations.append(exp[:8])

        # Sort and limit
        expirations = sorted(set(expirations))[:num_contracts]
        return expirations

    def get_front_month_symbol(self, symbol: str) -> Optional[str]:
        """
        Get front month contract symbol with expiry suffix.

        Args:
            symbol: Base symbol (ES, NQ, etc.)

        Returns:
            Full contract symbol (e.g., ESH24) or None
        """
        expirations = self.get_contract_expirations(symbol, num_contracts=1)
        if expirations:
            # Convert YYYYMMDD to contract month code
            exp = expirations[0]
            year = exp[2:4]
            month = int(exp[4:6])
            month_code = ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"][month - 1]
            return f"{symbol}{month_code}{year}"
        return None

    @staticmethod
    def _futures_type_to_market_type(futures_type: FuturesType) -> MarketType:
        """Convert FuturesType to MarketType."""
        mapping = {
            FuturesType.INDEX_FUTURES: MarketType.INDEX_FUTURES,
            FuturesType.COMMODITY_FUTURES: MarketType.COMMODITY_FUTURES,
            FuturesType.CURRENCY_FUTURES: MarketType.CURRENCY_FUTURES,
            FuturesType.BOND_FUTURES: MarketType.BOND_FUTURES,
        }
        return mapping.get(futures_type, MarketType.INDEX_FUTURES)
