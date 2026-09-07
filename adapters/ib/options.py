# -*- coding: utf-8 -*-
"""
adapters/ib/options.py
Interactive Brokers options market data and execution adapter.

Phase 2: US Exchange Adapters

Provides comprehensive options trading capabilities through IB TWS API:
- Option chain retrieval with caching
- Real-time option quotes and streaming
- Greeks calculation (via IB or local)
- Option order submission
- What-if margin calculations
- Combo/spread order support

IB Options Specifics:
    - Options use the Option contract type (not Future)
    - OCC symbology: SYMBOL(6) + YYMMDD + C/P + STRIKE(8)
    - Exchange codes: SMART (best execution), CBOE, ISE, PHLX, BOX, BATS
    - Greeks are provided by IB for US equity options

Rate Limits (handled by options_rate_limiter.py):
    - Option chains: 10/min (we use 8)
    - Quotes: 100/sec (we use 80)
    - Orders: 50/sec (we use 40)

References:
    - IB Options: https://interactivebrokers.github.io/tws-api/options.html
    - IB Margin: https://interactivebrokers.github.io/tws-api/margin.html
    - ib_insync options: https://ib-insync.readthedocs.io/api.html#options
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from adapters.base import MarketDataAdapter, OrderExecutionAdapter, OrderResult
from adapters.models import ExchangeVendor, AccountInfo

# Import from existing IB adapters
from adapters.ib.market_data import (
    IBMarketDataAdapter,
    IBConnectionManager,
    IBRateLimiter,
    IB_INSYNC_AVAILABLE,
)
from adapters.ib.order_execution import IBOrderExecutionAdapter
from adapters.ib.options_rate_limiter import (
    IBOptionsRateLimitManager,
    RequestPriority,
    create_options_rate_limiter,
)

# Import core options models
from core_options import (
    OptionsContractSpec,
    OptionType,
    ExerciseStyle,
    SettlementType,
    GreeksResult,
    OptionContract,
    OPTIONS_CONTRACT_MULTIPLIER,
)

# Import error handling
from core_errors import BotError

logger = logging.getLogger(__name__)

# Conditional imports for ib_insync
if IB_INSYNC_AVAILABLE:
    from ib_insync import (
        IB,
        Option,
        Index,
        Stock,
        Contract,
        MarketOrder,
        LimitOrder,
        Trade,
        Ticker,
        util,
    )
else:
    IB = None
    Option = None
    Index = None
    Stock = None
    Contract = None


# =============================================================================
# Options Quote Data
# =============================================================================


@dataclass
class OptionsQuote:
    """
    Real-time options quote with Greeks.

    Attributes:
        contract: Options contract specification
        bid: Bid price
        ask: Ask price
        last: Last traded price
        bid_size: Bid size (contracts)
        ask_size: Ask size (contracts)
        volume: Trading volume
        open_interest: Open interest
        underlying_price: Current underlying price
        greeks: Option Greeks from IB
        timestamp_ns: Quote timestamp in nanoseconds
    """

    contract: OptionsContractSpec
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    last: Optional[Decimal] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    underlying_price: Optional[Decimal] = None
    greeks: Optional[GreeksResult] = None
    timestamp_ns: int = 0

    @property
    def mid(self) -> Optional[Decimal]:
        """Calculate mid price."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return None

    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None

    @property
    def spread_bps(self) -> Optional[float]:
        """Calculate spread in basis points."""
        if self.mid is not None and self.spread is not None and self.mid > 0:
            return float(self.spread / self.mid) * 10000
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "contract": self.contract.to_dict(),
            "bid": str(self.bid) if self.bid else None,
            "ask": str(self.ask) if self.ask else None,
            "last": str(self.last) if self.last else None,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "underlying_price": str(self.underlying_price) if self.underlying_price else None,
            "greeks": self.greeks.to_dict() if self.greeks else None,
            "timestamp_ns": self.timestamp_ns,
        }


@dataclass
class OptionsChainData:
    """
    Complete options chain for an underlying/expiration.

    Attributes:
        underlying: Underlying symbol
        expiration: Expiration date
        calls: List of call options
        puts: List of put options
        underlying_price: Current underlying price
        timestamp_ns: Chain retrieval timestamp
    """

    underlying: str
    expiration: date
    calls: List[OptionsContractSpec] = field(default_factory=list)
    puts: List[OptionsContractSpec] = field(default_factory=list)
    underlying_price: Optional[Decimal] = None
    timestamp_ns: int = 0

    @property
    def all_contracts(self) -> List[OptionsContractSpec]:
        """Get all contracts (calls + puts)."""
        return self.calls + self.puts

    @property
    def strikes(self) -> List[Decimal]:
        """Get unique strike prices."""
        return sorted(set(c.strike for c in self.all_contracts))

    def get_atm_strike(self) -> Optional[Decimal]:
        """Get closest at-the-money strike."""
        if self.underlying_price is None or not self.strikes:
            return None
        return min(self.strikes, key=lambda k: abs(k - self.underlying_price))

    def filter_by_moneyness(
        self,
        spot: float,
        min_delta: float = 0.05,
        max_delta: float = 0.95,
    ) -> "OptionsChainData":
        """Filter contracts by approximate moneyness."""
        # Simple strike-based filtering (without Greeks)
        spot_d = Decimal(str(spot))
        lower_strike = spot_d * Decimal("0.7")  # ~30% OTM
        upper_strike = spot_d * Decimal("1.3")  # ~30% OTM

        return OptionsChainData(
            underlying=self.underlying,
            expiration=self.expiration,
            calls=[c for c in self.calls if lower_strike <= c.strike <= upper_strike],
            puts=[p for p in self.puts if lower_strike <= p.strike <= upper_strike],
            underlying_price=self.underlying_price,
            timestamp_ns=self.timestamp_ns,
        )


# =============================================================================
# Options Order
# =============================================================================


@dataclass
class OptionsOrder:
    """
    Options order specification.

    Attributes:
        contract: Options contract to trade
        side: BUY or SELL
        qty: Number of contracts
        order_type: MARKET, LIMIT, etc.
        limit_price: Limit price (for limit orders)
        time_in_force: DAY, GTC, IOC, FOK
        client_order_id: Client-assigned order ID
    """

    contract: OptionsContractSpec
    side: str  # BUY or SELL
    qty: int
    order_type: str = "LIMIT"  # MARKET, LIMIT
    limit_price: Optional[Decimal] = None
    time_in_force: str = "DAY"
    client_order_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate order."""
        if self.side.upper() not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side: {self.side}")
        if self.qty <= 0:
            raise ValueError(f"Quantity must be positive: {self.qty}")
        if self.order_type.upper() == "LIMIT" and self.limit_price is None:
            raise ValueError("Limit price required for LIMIT orders")


@dataclass
class OptionsOrderResult:
    """
    Result of options order submission.
    """

    success: bool
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    status: Optional[str] = None
    filled_qty: int = 0
    avg_fill_price: Optional[Decimal] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "status": self.status,
            "filled_qty": self.filled_qty,
            "avg_fill_price": str(self.avg_fill_price) if self.avg_fill_price else None,
            "error_message": self.error_message,
        }


@dataclass
class MarginRequirement:
    """
    Options margin requirement from what-if calculation.

    Attributes:
        initial_margin: Initial margin requirement
        maintenance_margin: Maintenance margin requirement
        commission: Estimated commission
        equity_impact: Impact on buying power
    """

    initial_margin: Decimal
    maintenance_margin: Decimal
    commission: Decimal = Decimal("0")
    equity_impact: Decimal = Decimal("0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "initial_margin": str(self.initial_margin),
            "maintenance_margin": str(self.maintenance_margin),
            "commission": str(self.commission),
            "equity_impact": str(self.equity_impact),
        }


# =============================================================================
# IB Options Market Data Adapter
# =============================================================================


class IBOptionsMarketDataAdapter(IBMarketDataAdapter):
    """
    IB options market data adapter.

    Extends IBMarketDataAdapter with options-specific functionality:
    - Option chain retrieval
    - Real-time option quotes with Greeks
    - Streaming option quotes
    - Underlying price tracking

    Configuration:
        host: TWS/Gateway host (default: 127.0.0.1)
        port: TWS port (7497 paper, 7496 live)
        client_id: Unique client ID
        default_exchange: Default options exchange (default: SMART)
        use_local_greeks: Calculate Greeks locally vs IB (default: False)
        rate_limiter_profile: Rate limiter profile (default: "default")

    Usage:
        adapter = IBOptionsMarketDataAdapter(
            vendor=ExchangeVendor.IB,
            config={"port": 7497}
        )
        adapter.connect()

        # Get option chain
        chain = adapter.get_option_chain("AAPL", date(2024, 12, 20))

        # Get quote with Greeks
        quote = adapter.get_option_quote(chain.calls[0])

        adapter.disconnect()
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.IB,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor, config)
        self._default_options_exchange = self._config.get("default_exchange", "SMART")
        self._use_local_greeks = self._config.get("use_local_greeks", False)

        # Options rate limiter
        rate_limiter_profile = self._config.get("rate_limiter_profile", "default")
        self._options_rate_limiter = create_options_rate_limiter(rate_limiter_profile)

        # Active underlying price subscriptions
        self._underlying_prices: Dict[str, Decimal] = {}
        self._underlying_tickers: Dict[str, Any] = {}  # IB Ticker objects

    # =========================================================================
    # Contract Creation
    # =========================================================================

    def _create_option_contract(
        self,
        underlying: str,
        expiration: date,
        strike: float,
        option_type: OptionType,
        exchange: Optional[str] = None,
    ) -> Any:
        """
        Create IB Option contract.

        Args:
            underlying: Underlying symbol
            expiration: Expiration date
            strike: Strike price
            option_type: CALL or PUT
            exchange: Exchange override

        Returns:
            IB Option contract object
        """
        if not IB_INSYNC_AVAILABLE:
            raise ImportError("ib_insync is required for IB options")

        right = "C" if option_type == OptionType.CALL else "P"
        expiry = expiration.strftime("%Y%m%d")

        return Option(
            underlying,
            expiry,
            strike,
            right,
            exchange=exchange or self._default_options_exchange,
        )

    def _create_underlying_contract(self, symbol: str) -> Any:
        """
        Create IB contract for underlying.

        Handles stocks and indices differently.
        """
        if not IB_INSYNC_AVAILABLE:
            raise ImportError("ib_insync is required for IB options")

        # Index symbols (no underlying stock)
        index_symbols = {"SPX", "NDX", "RUT", "VIX", "DJX"}

        if symbol in index_symbols:
            return Index(symbol, "CBOE")
        else:
            return Stock(symbol, "SMART", "USD")

    # =========================================================================
    # Option Chain
    # =========================================================================

    def get_option_chain(
        self,
        underlying: str,
        expiration: Optional[date] = None,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> List[OptionsChainData]:
        """
        Get option chains for underlying.

        Args:
            underlying: Underlying symbol (e.g., "AAPL")
            expiration: Specific expiration date (None = all available)
            use_cache: Use cached data if available
            force_refresh: Force refresh even if cached

        Returns:
            List of OptionsChainData for each expiration
        """
        if not self._conn_manager or not self._conn_manager.is_connected:
            raise ConnectionError("Not connected to IB")

        # Check cache first
        if use_cache and not force_refresh and expiration is not None:
            cached = self._options_rate_limiter.get_cached_chain(underlying, expiration)
            if cached is not None:
                return [cached]

        # Wait for rate limit slot
        if not self._options_rate_limiter.wait_for_chain_slot(timeout=60.0):
            raise TimeoutError("Rate limit timeout waiting for chain request slot")

        # Get underlying contract
        underlying_contract = self._create_underlying_contract(underlying)
        self._ib.qualifyContracts(underlying_contract)

        # Get current underlying price
        ticker = self._ib.reqMktData(underlying_contract, snapshot=True)
        self._ib.sleep(1)
        underlying_price = Decimal(str(ticker.last or ticker.close or 0))
        self._ib.cancelMktData(underlying_contract)

        # Request option chains
        chains_raw = self._ib.reqSecDefOptParams(
            underlying_contract.symbol,
            "",  # Empty = all exchanges
            underlying_contract.secType,
            underlying_contract.conId,
        )

        self._options_rate_limiter.record_chain_request()

        if not chains_raw:
            logger.warning(f"No option chains found for {underlying}")
            return []

        # Process chains
        result: List[OptionsChainData] = []

        for chain_def in chains_raw:
            # Filter expirations if specified
            for exp_str in chain_def.expirations:
                exp_date = datetime.strptime(exp_str, "%Y%m%d").date()

                if expiration is not None and exp_date != expiration:
                    continue

                # Build chain data
                calls = []
                puts = []

                for strike in chain_def.strikes:
                    # Create call spec
                    call_occ = self._make_occ_symbol(underlying, exp_date, strike, "C")
                    calls.append(
                        OptionsContractSpec(
                            symbol=call_occ,
                            underlying=underlying,
                            option_type=OptionType.CALL,
                            strike=Decimal(str(strike)),
                            expiration=exp_date,
                            exchange=chain_def.exchange,
                            multiplier=chain_def.multiplier,
                        )
                    )

                    # Create put spec
                    put_occ = self._make_occ_symbol(underlying, exp_date, strike, "P")
                    puts.append(
                        OptionsContractSpec(
                            symbol=put_occ,
                            underlying=underlying,
                            option_type=OptionType.PUT,
                            strike=Decimal(str(strike)),
                            expiration=exp_date,
                            exchange=chain_def.exchange,
                            multiplier=chain_def.multiplier,
                        )
                    )

                chain_data = OptionsChainData(
                    underlying=underlying,
                    expiration=exp_date,
                    calls=calls,
                    puts=puts,
                    underlying_price=underlying_price,
                    timestamp_ns=int(time.time() * 1e9),
                )

                # Cache the chain
                if use_cache:
                    self._options_rate_limiter.cache_chain(underlying, exp_date, chain_data)

                result.append(chain_data)

        return result

    def get_expirations(self, underlying: str) -> List[date]:
        """
        Get available expiration dates for underlying.

        Args:
            underlying: Underlying symbol

        Returns:
            List of available expiration dates
        """
        chains = self.get_option_chain(underlying)
        return sorted(set(c.expiration for c in chains))

    # =========================================================================
    # Option Quotes
    # =========================================================================

    def get_option_quote(
        self,
        contract: OptionsContractSpec,
        with_greeks: bool = True,
    ) -> OptionsQuote:
        """
        Get real-time quote for an option.

        Args:
            contract: Options contract specification
            with_greeks: Include Greeks in response

        Returns:
            OptionsQuote with current market data
        """
        if not self._conn_manager or not self._conn_manager.is_connected:
            raise ConnectionError("Not connected to IB")

        # Wait for rate limit
        if not self._options_rate_limiter.wait_for_quote_slot():
            raise TimeoutError("Rate limit timeout waiting for quote slot")

        # Create IB option contract
        ib_contract = self._create_option_contract(
            contract.underlying,
            contract.expiration,
            float(contract.strike),
            contract.option_type,
            contract.exchange,
        )
        self._ib.qualifyContracts(ib_contract)

        # Request market data
        generic_ticks = "106,104"  # 106=implied vol, 104=bid/ask/last
        ticker = self._ib.reqMktData(ib_contract, genericTickList=generic_ticks, snapshot=True)
        self._ib.sleep(1)  # Wait for data

        self._options_rate_limiter.record_quote_request()

        # Build quote
        quote = self._build_quote_from_ticker(contract, ticker, with_greeks)

        # Cancel subscription
        self._ib.cancelMktData(ib_contract)

        return quote

    def get_option_quotes_batch(
        self,
        contracts: List[OptionsContractSpec],
        with_greeks: bool = True,
    ) -> List[OptionsQuote]:
        """
        Get quotes for multiple options (batched for efficiency).

        Args:
            contracts: List of contracts to quote
            with_greeks: Include Greeks

        Returns:
            List of OptionsQuote objects
        """
        if not self._conn_manager or not self._conn_manager.is_connected:
            raise ConnectionError("Not connected to IB")

        if not contracts:
            return []

        # Subscribe to all contracts
        tickers = []
        ib_contracts = []

        for contract in contracts:
            # Rate limit check
            if not self._options_rate_limiter.wait_for_quote_slot(timeout=5.0):
                logger.warning(f"Rate limit timeout for {contract.symbol}")
                continue

            ib_contract = self._create_option_contract(
                contract.underlying,
                contract.expiration,
                float(contract.strike),
                contract.option_type,
                contract.exchange,
            )
            self._ib.qualifyContracts(ib_contract)

            ticker = self._ib.reqMktData(ib_contract, genericTickList="106,104", snapshot=True)
            tickers.append((contract, ticker))
            ib_contracts.append(ib_contract)
            self._options_rate_limiter.record_quote_request()

        # Wait for all data
        self._ib.sleep(2)

        # Build quotes
        quotes = []
        for contract, ticker in tickers:
            quote = self._build_quote_from_ticker(contract, ticker, with_greeks)
            quotes.append(quote)

        # Cancel all subscriptions
        for ib_contract in ib_contracts:
            self._ib.cancelMktData(ib_contract)

        return quotes

    def stream_option_quotes(
        self,
        contracts: List[OptionsContractSpec],
        with_greeks: bool = True,
    ) -> Iterator[OptionsQuote]:
        """
        Stream real-time option quotes.

        Args:
            contracts: List of contracts to stream
            with_greeks: Include Greeks in updates

        Yields:
            OptionsQuote as updates arrive
        """
        if not self._conn_manager or not self._conn_manager.is_connected:
            raise ConnectionError("Not connected to IB")

        # Subscribe to all contracts
        contract_map: Dict[int, Tuple[OptionsContractSpec, Any, Any]] = (
            {}
        )  # conId -> (spec, ib_contract, ticker)

        for contract in contracts:
            if not self._options_rate_limiter.add_subscription(contract.symbol):
                logger.warning(f"Subscription limit reached, skipping {contract.symbol}")
                continue

            ib_contract = self._create_option_contract(
                contract.underlying,
                contract.expiration,
                float(contract.strike),
                contract.option_type,
                contract.exchange,
            )
            self._ib.qualifyContracts(ib_contract)

            ticker = self._ib.reqMktData(ib_contract, genericTickList="106,104")
            contract_map[ib_contract.conId] = (contract, ib_contract, ticker)

        try:
            while True:
                self._ib.sleep(0.1)

                for con_id, (contract, ib_contract, ticker) in contract_map.items():
                    if ticker.bid is not None and ticker.ask is not None:
                        quote = self._build_quote_from_ticker(contract, ticker, with_greeks)
                        yield quote
        finally:
            # Cleanup
            for con_id, (contract, ib_contract, ticker) in contract_map.items():
                self._ib.cancelMktData(ib_contract)
                self._options_rate_limiter.remove_subscription(contract.symbol)

    async def stream_option_quotes_async(
        self,
        contracts: List[OptionsContractSpec],
        with_greeks: bool = True,
    ) -> AsyncIterator[OptionsQuote]:
        """
        Async stream of option quotes.

        Args:
            contracts: List of contracts to stream
            with_greeks: Include Greeks in updates

        Yields:
            OptionsQuote as updates arrive
        """
        if not self._conn_manager or not self._conn_manager.is_connected:
            raise ConnectionError("Not connected to IB")

        # Subscribe to all contracts
        contract_map: Dict[int, Tuple[OptionsContractSpec, Any, Any]] = {}

        for contract in contracts:
            if not self._options_rate_limiter.add_subscription(contract.symbol):
                logger.warning(f"Subscription limit reached, skipping {contract.symbol}")
                continue

            ib_contract = self._create_option_contract(
                contract.underlying,
                contract.expiration,
                float(contract.strike),
                contract.option_type,
                contract.exchange,
            )
            self._ib.qualifyContracts(ib_contract)

            ticker = self._ib.reqMktData(ib_contract, genericTickList="106,104")
            contract_map[ib_contract.conId] = (contract, ib_contract, ticker)

        try:
            while True:
                await asyncio.sleep(0.1)

                for con_id, (contract, ib_contract, ticker) in contract_map.items():
                    if ticker.bid is not None and ticker.ask is not None:
                        quote = self._build_quote_from_ticker(contract, ticker, with_greeks)
                        yield quote
        finally:
            # Cleanup
            for con_id, (contract, ib_contract, ticker) in contract_map.items():
                self._ib.cancelMktData(ib_contract)
                self._options_rate_limiter.remove_subscription(contract.symbol)

    # =========================================================================
    # Underlying Price
    # =========================================================================

    def get_underlying_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get current underlying price.

        Args:
            symbol: Underlying symbol

        Returns:
            Current price or None if not available
        """
        if symbol in self._underlying_prices:
            return self._underlying_prices[symbol]

        contract = self._create_underlying_contract(symbol)
        self._ib.qualifyContracts(contract)

        ticker = self._ib.reqMktData(contract, snapshot=True)
        self._ib.sleep(1)

        price = None
        if ticker.last is not None:
            price = Decimal(str(ticker.last))
        elif ticker.close is not None:
            price = Decimal(str(ticker.close))

        self._ib.cancelMktData(contract)

        if price is not None:
            self._underlying_prices[symbol] = price

        return price

    def subscribe_underlying(self, symbol: str) -> None:
        """
        Subscribe to underlying price updates.

        Args:
            symbol: Underlying symbol
        """
        if symbol in self._underlying_tickers:
            return

        contract = self._create_underlying_contract(symbol)
        self._ib.qualifyContracts(contract)

        ticker = self._ib.reqMktData(contract)
        self._underlying_tickers[symbol] = (contract, ticker)

    def unsubscribe_underlying(self, symbol: str) -> None:
        """
        Unsubscribe from underlying price updates.

        Args:
            symbol: Underlying symbol
        """
        if symbol not in self._underlying_tickers:
            return

        contract, ticker = self._underlying_tickers.pop(symbol)
        self._ib.cancelMktData(contract)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _make_occ_symbol(
        self,
        underlying: str,
        expiration: date,
        strike: float,
        opt_type: str,
    ) -> str:
        """Create OCC symbol from components."""
        symbol_padded = underlying.ljust(6)
        date_str = expiration.strftime("%y%m%d")
        strike_int = int(strike * 1000)
        strike_str = f"{strike_int:08d}"
        return f"{symbol_padded}{date_str}{opt_type}{strike_str}"

    def _build_quote_from_ticker(
        self,
        contract: OptionsContractSpec,
        ticker: Any,
        with_greeks: bool,
    ) -> OptionsQuote:
        """Build OptionsQuote from IB Ticker."""
        greeks = None

        if with_greeks and ticker.modelGreeks is not None:
            mg = ticker.modelGreeks
            greeks = GreeksResult(
                delta=mg.delta or 0.0,
                gamma=mg.gamma or 0.0,
                theta=mg.theta or 0.0,
                vega=mg.vega or 0.0,
                rho=0.0,  # IB doesn't provide rho in modelGreeks
                vanna=0.0,
                volga=0.0,
                charm=0.0,
                speed=0.0,
                color=0.0,
                zomma=0.0,
                ultima=0.0,
                timestamp_ns=int(time.time() * 1e9),
                spot=mg.undPrice,
                strike=float(contract.strike),
                time_to_expiry=None,
                volatility=mg.impliedVol,
            )

        return OptionsQuote(
            contract=contract,
            bid=Decimal(str(ticker.bid)) if ticker.bid is not None else None,
            ask=Decimal(str(ticker.ask)) if ticker.ask is not None else None,
            last=Decimal(str(ticker.last)) if ticker.last is not None else None,
            bid_size=int(ticker.bidSize) if ticker.bidSize else None,
            ask_size=int(ticker.askSize) if ticker.askSize else None,
            volume=int(ticker.volume) if ticker.volume else None,
            open_interest=None,  # Not in real-time data
            underlying_price=(
                Decimal(str(ticker.modelGreeks.undPrice)) if ticker.modelGreeks else None
            ),
            greeks=greeks,
            timestamp_ns=int(time.time() * 1e9),
        )

    # =========================================================================
    # Rate Limiter Access
    # =========================================================================

    @property
    def options_rate_limiter(self) -> IBOptionsRateLimitManager:
        """Get the options rate limiter."""
        return self._options_rate_limiter

    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return self._options_rate_limiter.get_stats()


# =============================================================================
# IB Options Order Execution Adapter
# =============================================================================


class IBOptionsOrderExecutionAdapter(IBOrderExecutionAdapter):
    """
    IB options order execution adapter.

    Extends IBOrderExecutionAdapter with options-specific functionality:
    - Option order submission
    - What-if margin calculations
    - Position Greeks aggregation
    - Exercise and assignment handling

    Configuration:
        host: TWS/Gateway host (default: 127.0.0.1)
        port: TWS port (7497 paper, 7496 live)
        client_id: Unique client ID
        default_exchange: Default options exchange (default: SMART)
        rate_limiter_profile: Rate limiter profile

    Usage:
        adapter = IBOptionsOrderExecutionAdapter(
            vendor=ExchangeVendor.IB,
            config={"port": 7497}
        )
        adapter.connect()

        # Submit option order
        order = OptionsOrder(
            contract=call_option,
            side="BUY",
            qty=1,
            order_type="LIMIT",
            limit_price=Decimal("2.50"),
        )
        result = adapter.submit_option_order(order)

        adapter.disconnect()
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.IB,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor, config)
        self._default_options_exchange = self._config.get("default_exchange", "SMART")

        # Options rate limiter
        rate_limiter_profile = self._config.get("rate_limiter_profile", "default")
        self._options_rate_limiter = create_options_rate_limiter(rate_limiter_profile)

    # =========================================================================
    # Contract Creation
    # =========================================================================

    def _create_option_contract(
        self,
        contract: OptionsContractSpec,
    ) -> Any:
        """Create IB Option contract from spec."""
        if not IB_INSYNC_AVAILABLE:
            raise ImportError("ib_insync is required for IB options")

        right = "C" if contract.option_type == OptionType.CALL else "P"
        expiry = contract.expiration.strftime("%Y%m%d")

        return Option(
            contract.underlying,
            expiry,
            float(contract.strike),
            right,
            exchange=contract.exchange or self._default_options_exchange,
        )

    # =========================================================================
    # Order Submission
    # =========================================================================

    def submit_option_order(self, order: OptionsOrder) -> OptionsOrderResult:
        """
        Submit option order.

        Args:
            order: Options order to submit

        Returns:
            OptionsOrderResult with execution details
        """
        if not self._ib or not self._ib.isConnected():
            return OptionsOrderResult(
                success=False,
                error_message="Not connected to IB",
            )

        # Wait for rate limit
        if not self._options_rate_limiter.wait_for_order_slot():
            return OptionsOrderResult(
                success=False,
                error_message="Rate limit timeout",
            )

        try:
            # Create IB contract
            ib_contract = self._create_option_contract(order.contract)
            self._ib.qualifyContracts(ib_contract)

            # Create IB order
            action = order.side.upper()

            if order.order_type.upper() == "MARKET":
                ib_order = MarketOrder(action, order.qty)
            else:
                ib_order = LimitOrder(action, order.qty, float(order.limit_price))

            ib_order.tif = order.time_in_force.upper()

            if order.client_order_id:
                ib_order.orderRef = order.client_order_id

            # Place order
            trade = self._ib.placeOrder(ib_contract, ib_order)

            self._options_rate_limiter.record_order_request()

            # Wait briefly for order status
            self._ib.sleep(0.5)

            return OptionsOrderResult(
                success=True,
                order_id=str(trade.order.orderId),
                client_order_id=trade.order.orderRef or str(trade.order.orderId),
                status=trade.orderStatus.status,
                filled_qty=int(trade.orderStatus.filled),
                avg_fill_price=(
                    Decimal(str(trade.orderStatus.avgFillPrice))
                    if trade.orderStatus.avgFillPrice
                    else None
                ),
            )

        except Exception as e:
            logger.error(f"Option order submission failed: {e}")
            return OptionsOrderResult(
                success=False,
                error_message=str(e),
            )

    def submit_option_market_order(
        self,
        contract: OptionsContractSpec,
        side: str,
        qty: int,
    ) -> OptionsOrderResult:
        """
        Submit market order for option.

        Args:
            contract: Options contract
            side: BUY or SELL
            qty: Number of contracts

        Returns:
            OptionsOrderResult
        """
        order = OptionsOrder(
            contract=contract,
            side=side,
            qty=qty,
            order_type="MARKET",
        )
        return self.submit_option_order(order)

    def submit_option_limit_order(
        self,
        contract: OptionsContractSpec,
        side: str,
        qty: int,
        limit_price: Decimal,
        time_in_force: str = "DAY",
    ) -> OptionsOrderResult:
        """
        Submit limit order for option.

        Args:
            contract: Options contract
            side: BUY or SELL
            qty: Number of contracts
            limit_price: Limit price
            time_in_force: DAY, GTC, IOC

        Returns:
            OptionsOrderResult
        """
        order = OptionsOrder(
            contract=contract,
            side=side,
            qty=qty,
            order_type="LIMIT",
            limit_price=limit_price,
            time_in_force=time_in_force,
        )
        return self.submit_option_order(order)

    # =========================================================================
    # Margin Calculation
    # =========================================================================

    def get_option_margin_requirement(
        self,
        contract: OptionsContractSpec,
        side: str,
        qty: int,
    ) -> MarginRequirement:
        """
        Get margin requirement for option order (what-if).

        Args:
            contract: Options contract
            side: BUY or SELL
            qty: Number of contracts

        Returns:
            MarginRequirement with initial/maintenance margin
        """
        if not self._ib or not self._ib.isConnected():
            raise ConnectionError("Not connected to IB")

        ib_contract = self._create_option_contract(contract)
        self._ib.qualifyContracts(ib_contract)

        action = "BUY" if side.upper() == "BUY" else "SELL"
        order = MarketOrder(action, qty)

        what_if = self._ib.whatIfOrder(ib_contract, order)

        return MarginRequirement(
            initial_margin=Decimal(str(what_if.initMarginChange or 0)),
            maintenance_margin=Decimal(str(what_if.maintMarginChange or 0)),
            commission=Decimal(str(what_if.commission or 0)),
            equity_impact=Decimal(str(what_if.equityWithLoanChange or 0)),
        )

    # =========================================================================
    # Position Management
    # =========================================================================

    def get_option_positions(self) -> List[Dict[str, Any]]:
        """
        Get all option positions.

        Returns:
            List of option positions with details
        """
        if not self._ib or not self._ib.isConnected():
            raise ConnectionError("Not connected to IB")

        positions = []

        for pos in self._ib.positions():
            contract = pos.contract
            if hasattr(contract, "secType") and contract.secType == "OPT":
                qty = Decimal(str(pos.position))
                if qty == 0:
                    continue

                positions.append(
                    {
                        "symbol": contract.symbol,
                        "underlying": contract.symbol,
                        "strike": Decimal(str(contract.strike)),
                        "expiration": datetime.strptime(
                            contract.lastTradeDateOrContractMonth, "%Y%m%d"
                        ).date(),
                        "option_type": OptionType.CALL if contract.right == "C" else OptionType.PUT,
                        "qty": qty,
                        "avg_cost": Decimal(str(pos.avgCost)),
                        "exchange": contract.exchange,
                    }
                )

        return positions

    def cancel_option_order(
        self,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> bool:
        """
        Cancel an open option order.

        Args:
            order_id: Exchange order ID
            client_order_id: Client order ID

        Returns:
            True if cancelled successfully
        """
        return self.cancel_order(order_id, client_order_id)

    # =========================================================================
    # Rate Limiter
    # =========================================================================

    @property
    def options_rate_limiter(self) -> IBOptionsRateLimitManager:
        """Get the options rate limiter."""
        return self._options_rate_limiter


# =============================================================================
# Factory Functions
# =============================================================================


def create_ib_options_market_data_adapter(
    config: Optional[Dict[str, Any]] = None,
) -> IBOptionsMarketDataAdapter:
    """
    Create IB options market data adapter.

    Args:
        config: Optional configuration override

    Returns:
        Configured IBOptionsMarketDataAdapter
    """
    return IBOptionsMarketDataAdapter(
        vendor=ExchangeVendor.IB,
        config=config or {},
    )


def create_ib_options_execution_adapter(
    config: Optional[Dict[str, Any]] = None,
) -> IBOptionsOrderExecutionAdapter:
    """
    Create IB options execution adapter.

    Args:
        config: Optional configuration override

    Returns:
        Configured IBOptionsOrderExecutionAdapter
    """
    return IBOptionsOrderExecutionAdapter(
        vendor=ExchangeVendor.IB,
        config=config or {},
    )


# =============================================================================
# Exports
# =============================================================================

# =============================================================================
# OCC Symbology Utilities
# =============================================================================


def parse_occ_symbol(occ_symbol: str) -> Dict[str, Any]:
    """
    Parse OCC option symbol to components.

    OCC Format: SYMBOL(6 chars, space-padded) + YYMMDD + C/P + STRIKE(8 digits)
    Example: "AAPL  241220C00200000" -> AAPL Dec 20 2024 $200 Call

    Args:
        occ_symbol: OCC-format option symbol

    Returns:
        Dict with symbol, expiration, option_type, strike
    """
    if len(occ_symbol) != 21:
        raise ValueError(f"Invalid OCC symbol length: {len(occ_symbol)}")

    symbol = occ_symbol[:6].strip()
    year = 2000 + int(occ_symbol[6:8])
    month = int(occ_symbol[8:10])
    day = int(occ_symbol[10:12])
    option_type = occ_symbol[12]
    strike = Decimal(occ_symbol[13:21]) / 1000

    return {
        "symbol": symbol,
        "expiration": date(year, month, day),
        "option_type": option_type,
        "strike": strike,
    }


def create_occ_symbol(
    underlying: str,
    expiration: date,
    option_type: str,
    strike: Decimal,
) -> str:
    """
    Create OCC option symbol from components.

    Args:
        underlying: Underlying symbol (e.g., "AAPL")
        expiration: Expiration date
        option_type: "C" for call, "P" for put
        strike: Strike price

    Returns:
        OCC-format symbol (21 chars)
    """
    # Pad symbol to 6 chars
    padded_symbol = underlying.ljust(6)[:6]
    # Format date as YYMMDD
    date_str = expiration.strftime("%y%m%d")
    # Format strike as 8-digit integer (strike * 1000)
    strike_int = int(strike * 1000)
    strike_str = f"{strike_int:08d}"
    return f"{padded_symbol}{date_str}{option_type}{strike_str}"


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# Alias for tests expecting IBOptionsAdapter
IBOptionsAdapter = IBOptionsMarketDataAdapter

# Alias dataclass names
IBOptionContract = OptionsContractSpec
IBOptionQuote = OptionsQuote
IBOptionGreeks = GreeksResult
IBOptionOrderResult = OptionsOrderResult


__all__ = [
    # Quote types
    "OptionsQuote",
    "OptionsChainData",
    # Order types
    "OptionsOrder",
    "OptionsOrderResult",
    "MarginRequirement",
    # Adapters
    "IBOptionsMarketDataAdapter",
    "IBOptionsOrderExecutionAdapter",
    # Backward compatibility aliases
    "IBOptionsAdapter",
    "IBOptionContract",
    "IBOptionQuote",
    "IBOptionGreeks",
    "IBOptionOrderResult",
    # OCC utilities
    "parse_occ_symbol",
    "create_occ_symbol",
    # Factory functions
    "create_ib_options_market_data_adapter",
    "create_ib_options_execution_adapter",
]
