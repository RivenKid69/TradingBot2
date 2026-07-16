# -*- coding: utf-8 -*-
"""
adapters/theta_data/options.py
Theta Data options data adapter for US equity options.

Phase 2: US Exchange Adapters

Theta Data provides comprehensive US options data at $100/month:
- Full US options universe (all listed options)
- Historical data back to 2013
- Real-time with 15-min delay (free tier)
- Real-time quotes (paid subscription)
- Greeks and IV calculations
- End-of-day chain snapshots

API Reference: https://www.thetadata.io/docs/

Data Types:
- Option Chains: Expirations, strikes, Greeks
- Quotes: NBBO with size, IV, Greeks
- Trades: Tick-level executions
- EOD: End-of-day snapshots
- Greeks: Delta, Gamma, Theta, Vega, Rho

Rate Limits:
- Free tier: 100 requests/min
- Pro tier: 1000 requests/min
- Enterprise: Unlimited

Note: This is a DATA-ONLY adapter (no order execution).
For execution, use IB or Alpaca.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import pandas as pd

from adapters.base import MarketDataAdapter
from adapters.models import ExchangeVendor
from core_models import Bar  # Bar is a core model, not an adapters.models symbol
from core_options import (
    OptionsContractSpec,
    OptionType,
    ExerciseStyle,
    SettlementType,
    GreeksResult,
    OPTIONS_CONTRACT_MULTIPLIER,
)

logger = logging.getLogger(__name__)

# Try to import thetadata library
try:
    from thetadata import ThetaClient
    THETA_DATA_AVAILABLE = True
except ImportError:
    ThetaClient = None
    THETA_DATA_AVAILABLE = False
    logger.info("thetadata not installed. Install with: pip install thetadata")


# =============================================================================
# Constants
# =============================================================================

# Theta Data API endpoints
THETA_DATA_BASE_URL = "https://api.thetadata.io"
THETA_DATA_WS_URL = "wss://api.thetadata.io/ws"

# Rate limit defaults
DEFAULT_RATE_LIMIT_PER_MIN = 100  # Free tier
PRO_RATE_LIMIT_PER_MIN = 1000

# Data intervals
VALID_INTERVALS = ["1min", "5min", "15min", "30min", "1h", "1d"]


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ThetaDataConfig:
    """
    Theta Data adapter configuration.

    Attributes:
        api_key: Theta Data API key (optional for some endpoints)
        username: Theta Data username
        password: Theta Data password (for terminal API)
        use_terminal: Use terminal API instead of REST
        timeout_sec: Request timeout in seconds
        rate_limit_per_min: Rate limit (depends on subscription)
        cache_ttl_sec: Cache TTL for chain data
        use_15min_delay: Use free 15-min delayed data
    """
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    use_terminal: bool = False
    timeout_sec: float = 30.0
    rate_limit_per_min: int = DEFAULT_RATE_LIMIT_PER_MIN
    cache_ttl_sec: float = 300.0  # 5 minutes
    use_15min_delay: bool = True


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ThetaDataQuote:
    """
    Theta Data options quote.

    Attributes:
        contract: Options contract specification
        bid: Bid price
        ask: Ask price
        bid_size: Bid size
        ask_size: Ask size
        mid: Mid price
        last: Last trade price
        volume: Volume
        open_interest: Open interest
        iv: Implied volatility
        delta: Option delta
        gamma: Option gamma
        theta: Option theta
        vega: Option vega
        rho: Option rho
        underlying_price: Underlying price
        timestamp: Quote timestamp
    """
    contract: OptionsContractSpec
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    mid: Optional[Decimal] = None
    last: Optional[Decimal] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    iv: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    underlying_price: Optional[Decimal] = None
    timestamp: Optional[datetime] = None

    @property
    def spread(self) -> Optional[Decimal]:
        """Bid-ask spread."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None

    @property
    def spread_bps(self) -> Optional[float]:
        """Spread in basis points of mid price."""
        if self.spread is not None and self.mid is not None and self.mid > 0:
            return float(self.spread / self.mid) * 10000
        return None

    def to_greeks(self) -> Optional[GreeksResult]:
        """Convert to GreeksResult."""
        if self.delta is None:
            return None
        return GreeksResult(
            delta=self.delta,
            gamma=self.gamma or 0.0,
            theta=self.theta or 0.0,
            vega=self.vega or 0.0,
            rho=self.rho or 0.0,
        )


@dataclass
class ThetaDataTrade:
    """
    Theta Data options trade.

    Attributes:
        contract: Options contract specification
        price: Trade price
        size: Trade size (contracts)
        timestamp: Trade timestamp
        exchange: Exchange code
        condition: Trade condition
    """
    contract: OptionsContractSpec
    price: Decimal
    size: int
    timestamp: datetime
    exchange: Optional[str] = None
    condition: Optional[str] = None


@dataclass
class ThetaDataChain:
    """
    Theta Data option chain.

    Attributes:
        underlying: Underlying symbol
        quotes: List of option quotes
        timestamp: Chain snapshot timestamp
    """
    underlying: str
    quotes: List[ThetaDataQuote] = field(default_factory=list)
    timestamp: Optional[datetime] = None

    @property
    def expirations(self) -> List[date]:
        """Get unique expiration dates."""
        return sorted(set(q.contract.expiration for q in self.quotes))

    @property
    def strikes(self) -> List[Decimal]:
        """Get unique strikes."""
        return sorted(set(q.contract.strike for q in self.quotes))

    def get_calls(self, expiration: Optional[date] = None) -> List[ThetaDataQuote]:
        """Get call quotes."""
        calls = [q for q in self.quotes if q.contract.option_type == OptionType.CALL]
        if expiration:
            calls = [q for q in calls if q.contract.expiration == expiration]
        return calls

    def get_puts(self, expiration: Optional[date] = None) -> List[ThetaDataQuote]:
        """Get put quotes."""
        puts = [q for q in self.quotes if q.contract.option_type == OptionType.PUT]
        if expiration:
            puts = [q for q in puts if q.contract.expiration == expiration]
        return puts


# =============================================================================
# Theta Data Options Adapter
# =============================================================================

class ThetaDataOptionsAdapter(MarketDataAdapter):
    """
    Theta Data options data adapter.

    Provides access to US options market data including:
    - Option chains with Greeks
    - Historical quotes (OHLC + Greeks + IV)
    - Tick-level trades
    - End-of-day snapshots

    Cost: $100/month (vs OPRA $2,500/month)

    Note: This is DATA-ONLY - no order execution.

    Example:
        adapter = ThetaDataOptionsAdapter(config=ThetaDataConfig(
            username="your_username",
            password="your_password",
        ))

        # Get option chain
        chain = adapter.get_option_chain("AAPL")

        # Get historical quotes
        quotes = adapter.get_historical_quotes(
            contract,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 1),
        )
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.THETA_DATA,
        config: Optional[Union[ThetaDataConfig, Mapping[str, Any]]] = None,
    ) -> None:
        """
        Initialize Theta Data adapter.

        Args:
            vendor: Exchange vendor enum
            config: Configuration (ThetaDataConfig or dict)
        """
        super().__init__(vendor, config if isinstance(config, Mapping) else None)

        if isinstance(config, ThetaDataConfig):
            self._config_obj = config
        else:
            self._config_obj = ThetaDataConfig(
                api_key=self._config.get("api_key"),
                username=self._config.get("username"),
                password=self._config.get("password"),
                use_terminal=self._config.get("use_terminal", False),
                timeout_sec=self._config.get("timeout_sec", 30.0),
                rate_limit_per_min=self._config.get("rate_limit_per_min", DEFAULT_RATE_LIMIT_PER_MIN),
                cache_ttl_sec=self._config.get("cache_ttl_sec", 300.0),
                use_15min_delay=self._config.get("use_15min_delay", True),
            )

        self._client = None
        self._session = None
        self._last_request_time = 0.0
        self._request_count = 0
        self._minute_start = time.time()

        # Cache for chain data
        self._chain_cache: Dict[str, Tuple[float, ThetaDataChain]] = {}

    def _ensure_rate_limit(self) -> None:
        """Ensure we don't exceed rate limits."""
        now = time.time()

        # Reset counter every minute
        if now - self._minute_start >= 60:
            self._request_count = 0
            self._minute_start = now

        # Wait if at limit
        if self._request_count >= self._config_obj.rate_limit_per_min:
            sleep_time = 60 - (now - self._minute_start)
            if sleep_time > 0:
                logger.debug(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self._request_count = 0
                self._minute_start = time.time()

        self._request_count += 1

    def _get_client(self):
        """Get or create Theta Data client."""
        if self._client is not None:
            return self._client

        if not THETA_DATA_AVAILABLE:
            raise ImportError(
                "thetadata not installed. Install with: pip install thetadata"
            )

        try:
            self._client = ThetaClient(
                username=self._config_obj.username,
                passwd=self._config_obj.password,
            )
            return self._client
        except Exception as e:
            logger.error(f"Failed to create Theta Data client: {e}")
            raise

    def _occ_to_contract(self, occ_symbol: str) -> OptionsContractSpec:
        """
        Convert OCC symbol to OptionsContractSpec.

        OCC format: SYMBOL(6) + YYMMDD + C/P + STRIKE(8)
        Example: AAPL  241220C00200000

        Args:
            occ_symbol: OCC formatted symbol

        Returns:
            OptionsContractSpec
        """
        # Parse OCC symbol (21 characters)
        if len(occ_symbol) < 21:
            raise ValueError(f"Invalid OCC symbol: {occ_symbol}")

        # Extract components
        underlying = occ_symbol[:6].strip()
        date_str = occ_symbol[6:12]
        opt_type = occ_symbol[12]
        strike_str = occ_symbol[13:21]

        # Parse date
        expiration = datetime.strptime(date_str, "%y%m%d").date()

        # Parse strike (8 digits: 5 whole + 3 decimal)
        strike = Decimal(strike_str) / 1000

        # Option type
        option_type = OptionType.CALL if opt_type == "C" else OptionType.PUT

        return OptionsContractSpec(
            underlying=underlying,
            expiration=expiration,
            strike=strike,
            option_type=option_type,
        )

    def _contract_to_occ(self, contract: OptionsContractSpec) -> str:
        """
        Convert OptionsContractSpec to OCC symbol.

        Args:
            contract: Options contract specification

        Returns:
            OCC formatted symbol
        """
        underlying = contract.underlying.ljust(6)
        date_str = contract.expiration.strftime("%y%m%d")
        opt_char = "C" if contract.option_type == OptionType.CALL else "P"
        strike_int = int(contract.strike * 1000)
        strike_str = f"{strike_int:08d}"

        return f"{underlying}{date_str}{opt_char}{strike_str}"

    # =========================================================================
    # Option Chain Methods
    # =========================================================================

    def get_expirations(self, underlying: str) -> List[date]:
        """
        Get available expiration dates for an underlying.

        Args:
            underlying: Underlying symbol

        Returns:
            List of expiration dates
        """
        self._ensure_rate_limit()

        try:
            client = self._get_client()

            # Use thetadata API to get expirations
            exps = client.get_expirations(underlying.upper())

            # Convert to date objects
            return [
                datetime.strptime(str(exp), "%Y%m%d").date()
                for exp in exps
            ]

        except Exception as e:
            logger.error(f"Failed to get expirations for {underlying}: {e}")
            return []

    def get_strikes(
        self,
        underlying: str,
        expiration: date,
    ) -> List[Decimal]:
        """
        Get available strike prices for an expiration.

        Args:
            underlying: Underlying symbol
            expiration: Expiration date

        Returns:
            List of strike prices
        """
        self._ensure_rate_limit()

        try:
            client = self._get_client()

            # Format date for API
            exp_str = expiration.strftime("%Y%m%d")

            # Get strikes
            strikes = client.get_strikes(
                underlying.upper(),
                exp_str,
            )

            return [Decimal(str(s)) for s in strikes]

        except Exception as e:
            logger.error(f"Failed to get strikes for {underlying} {expiration}: {e}")
            return []

    def get_option_chain(
        self,
        underlying: str,
        expiration: Optional[date] = None,
        use_cache: bool = True,
    ) -> ThetaDataChain:
        """
        Get full option chain for an underlying.

        Args:
            underlying: Underlying symbol
            expiration: Specific expiration (None = all)
            use_cache: Use cached data if available

        Returns:
            ThetaDataChain with quotes
        """
        cache_key = f"{underlying}:{expiration or 'all'}"

        # Check cache
        if use_cache and cache_key in self._chain_cache:
            cache_time, cached_chain = self._chain_cache[cache_key]
            if time.time() - cache_time < self._config_obj.cache_ttl_sec:
                return cached_chain

        self._ensure_rate_limit()

        try:
            client = self._get_client()

            quotes = []

            # Get expirations
            if expiration:
                expirations = [expiration]
            else:
                expirations = self.get_expirations(underlying)

            for exp in expirations:
                exp_str = exp.strftime("%Y%m%d")

                # Get quotes for both calls and puts
                for right in ["C", "P"]:
                    try:
                        self._ensure_rate_limit()

                        # Get at-the-money region quotes
                        data = client.get_quotes(
                            root=underlying.upper(),
                            exp=exp_str,
                            right=right,
                        )

                        if data is not None and not data.empty:
                            for _, row in data.iterrows():
                                contract = OptionsContractSpec(
                                    underlying=underlying.upper(),
                                    expiration=exp,
                                    strike=Decimal(str(row.get("strike", 0))),
                                    option_type=OptionType.CALL if right == "C" else OptionType.PUT,
                                )

                                quote = ThetaDataQuote(
                                    contract=contract,
                                    bid=Decimal(str(row.get("bid", 0))) if pd.notna(row.get("bid")) else None,
                                    ask=Decimal(str(row.get("ask", 0))) if pd.notna(row.get("ask")) else None,
                                    bid_size=int(row.get("bid_size", 0)) if pd.notna(row.get("bid_size")) else None,
                                    ask_size=int(row.get("ask_size", 0)) if pd.notna(row.get("ask_size")) else None,
                                    last=Decimal(str(row.get("last", 0))) if pd.notna(row.get("last")) else None,
                                    volume=int(row.get("volume", 0)) if pd.notna(row.get("volume")) else None,
                                    open_interest=int(row.get("open_interest", 0)) if pd.notna(row.get("open_interest")) else None,
                                    iv=float(row.get("iv", 0)) if pd.notna(row.get("iv")) else None,
                                    delta=float(row.get("delta", 0)) if pd.notna(row.get("delta")) else None,
                                    gamma=float(row.get("gamma", 0)) if pd.notna(row.get("gamma")) else None,
                                    theta=float(row.get("theta", 0)) if pd.notna(row.get("theta")) else None,
                                    vega=float(row.get("vega", 0)) if pd.notna(row.get("vega")) else None,
                                    underlying_price=Decimal(str(row.get("underlying_price", 0))) if pd.notna(row.get("underlying_price")) else None,
                                    timestamp=datetime.now(),
                                )

                                # Calculate mid
                                if quote.bid is not None and quote.ask is not None:
                                    quote.mid = (quote.bid + quote.ask) / 2

                                quotes.append(quote)

                    except Exception as e:
                        logger.warning(f"Failed to get {right} quotes for {underlying} {exp}: {e}")
                        continue

            chain = ThetaDataChain(
                underlying=underlying.upper(),
                quotes=quotes,
                timestamp=datetime.now(),
            )

            # Update cache
            self._chain_cache[cache_key] = (time.time(), chain)

            return chain

        except Exception as e:
            logger.error(f"Failed to get option chain for {underlying}: {e}")
            return ThetaDataChain(underlying=underlying.upper())

    # =========================================================================
    # Historical Data Methods
    # =========================================================================

    def get_historical_quotes(
        self,
        contract: OptionsContractSpec,
        start_date: date,
        end_date: date,
        interval: str = "1min",
    ) -> pd.DataFrame:
        """
        Get historical OHLCV quotes with Greeks and IV.

        Args:
            contract: Options contract specification
            start_date: Start date
            end_date: End date
            interval: Time interval (1min, 5min, 15min, 30min, 1h, 1d)

        Returns:
            DataFrame with columns:
            - timestamp, open, high, low, close, volume
            - bid, ask, mid
            - iv, delta, gamma, theta, vega
            - underlying_price
        """
        if interval not in VALID_INTERVALS:
            raise ValueError(f"Invalid interval: {interval}. Must be one of {VALID_INTERVALS}")

        self._ensure_rate_limit()

        try:
            client = self._get_client()

            # Format parameters
            root = contract.underlying.upper()
            exp_str = contract.expiration.strftime("%Y%m%d")
            strike = float(contract.strike)
            right = "C" if contract.option_type == OptionType.CALL else "P"
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")

            # Get OHLC data
            data = client.get_hist_stock_option(
                root=root,
                exp=exp_str,
                strike=strike,
                right=right,
                start_date=start_str,
                end_date=end_str,
                interval_size=int(interval.replace("min", "").replace("h", "60").replace("d", "1440")),
            )

            if data is None or data.empty:
                return pd.DataFrame()

            # Standardize column names
            rename_map = {
                "date": "timestamp",
                "ms_of_day": "ms_of_day",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
                "count": "trade_count",
            }

            df = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})

            # Add contract info
            df["underlying"] = root
            df["expiration"] = contract.expiration
            df["strike"] = float(contract.strike)
            df["option_type"] = contract.option_type.value

            return df

        except Exception as e:
            logger.error(f"Failed to get historical quotes for {contract}: {e}")
            return pd.DataFrame()

    def get_historical_trades(
        self,
        contract: OptionsContractSpec,
        trade_date: date,
    ) -> pd.DataFrame:
        """
        Get tick-level trade data.

        Args:
            contract: Options contract specification
            trade_date: Date to fetch trades for

        Returns:
            DataFrame with columns:
            - timestamp, price, size, exchange, condition
        """
        self._ensure_rate_limit()

        try:
            client = self._get_client()

            # Format parameters
            root = contract.underlying.upper()
            exp_str = contract.expiration.strftime("%Y%m%d")
            strike = float(contract.strike)
            right = "C" if contract.option_type == OptionType.CALL else "P"
            date_str = trade_date.strftime("%Y%m%d")

            # Get trade data
            data = client.get_trades(
                root=root,
                exp=exp_str,
                strike=strike,
                right=right,
                date=date_str,
            )

            if data is None or data.empty:
                return pd.DataFrame()

            # Standardize columns
            rename_map = {
                "ms_of_day": "timestamp_ms",
                "price": "price",
                "size": "size",
                "exchange": "exchange",
                "conditions": "condition",
            }

            df = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})

            # Convert ms to datetime
            if "timestamp_ms" in df.columns:
                df["timestamp"] = pd.to_datetime(
                    trade_date.strftime("%Y-%m-%d")
                ) + pd.to_timedelta(df["timestamp_ms"], unit="ms")

            return df

        except Exception as e:
            logger.error(f"Failed to get trades for {contract} on {trade_date}: {e}")
            return pd.DataFrame()

    def get_eod_chain(
        self,
        underlying: str,
        snapshot_date: date,
    ) -> pd.DataFrame:
        """
        Get end-of-day snapshot of full option chain.

        Args:
            underlying: Underlying symbol
            snapshot_date: Date for EOD snapshot

        Returns:
            DataFrame with full chain data including:
            - All expirations and strikes
            - Close prices, IV, Greeks
            - Open interest
        """
        self._ensure_rate_limit()

        try:
            client = self._get_client()

            date_str = snapshot_date.strftime("%Y%m%d")

            # Get EOD data
            data = client.get_eod_report(
                root=underlying.upper(),
                date=date_str,
            )

            if data is None or data.empty:
                return pd.DataFrame()

            # Add underlying column
            data["underlying"] = underlying.upper()

            return data

        except Exception as e:
            logger.error(f"Failed to get EOD chain for {underlying} on {snapshot_date}: {e}")
            return pd.DataFrame()

    # =========================================================================
    # Real-time/Streaming Methods
    # =========================================================================

    def get_quote(
        self,
        contract: OptionsContractSpec,
    ) -> Optional[ThetaDataQuote]:
        """
        Get current quote for a single contract.

        Note: May be 15-min delayed on free tier.

        Args:
            contract: Options contract specification

        Returns:
            ThetaDataQuote or None
        """
        chain = self.get_option_chain(
            underlying=contract.underlying,
            expiration=contract.expiration,
        )

        # Find matching quote
        for quote in chain.quotes:
            if (quote.contract.strike == contract.strike and
                quote.contract.option_type == contract.option_type):
                return quote

        return None

    def get_quotes_batch(
        self,
        contracts: List[OptionsContractSpec],
    ) -> List[ThetaDataQuote]:
        """
        Get quotes for multiple contracts.

        Args:
            contracts: List of options contracts

        Returns:
            List of ThetaDataQuotes
        """
        quotes = []

        # Group by underlying and expiration for efficiency
        groups: Dict[Tuple[str, date], List[OptionsContractSpec]] = {}
        for contract in contracts:
            key = (contract.underlying, contract.expiration)
            if key not in groups:
                groups[key] = []
            groups[key].append(contract)

        # Fetch chains and match
        for (underlying, expiration), group_contracts in groups.items():
            chain = self.get_option_chain(underlying, expiration)

            for contract in group_contracts:
                for quote in chain.quotes:
                    if (quote.contract.strike == contract.strike and
                        quote.contract.option_type == contract.option_type):
                        quotes.append(quote)
                        break

        return quotes

    # =========================================================================
    # Underlying Data Methods
    # =========================================================================

    def get_underlying_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get current underlying price.

        Args:
            symbol: Underlying symbol

        Returns:
            Current price or None
        """
        self._ensure_rate_limit()

        try:
            client = self._get_client()

            # Get stock quote
            data = client.get_stock_quotes(symbol.upper())

            if data is not None and not data.empty:
                row = data.iloc[-1]
                bid = row.get("bid", 0)
                ask = row.get("ask", 0)
                if bid > 0 and ask > 0:
                    return Decimal(str((bid + ask) / 2))
                last = row.get("last", 0)
                if last > 0:
                    return Decimal(str(last))

            return None

        except Exception as e:
            logger.error(f"Failed to get underlying price for {symbol}: {e}")
            return None

    # =========================================================================
    # MarketDataAdapter Interface Implementation
    # =========================================================================

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Bar]:
        """
        Get historical bars (for underlying or OCC option symbol).

        Args:
            symbol: Symbol (stock or OCC option symbol)
            timeframe: Timeframe (1m, 5m, 15m, 30m, 1h, 1d)
            limit: Maximum number of bars
            start: Start datetime
            end: End datetime

        Returns:
            List of Bar objects
        """
        # Check if this is an OCC option symbol
        if len(symbol) == 21 and symbol[12] in ("C", "P"):
            # This is an option
            contract = self._occ_to_contract(symbol)

            end_date = end.date() if end else date.today()
            start_date = start.date() if start else end_date - timedelta(days=30)

            # Convert timeframe
            interval_map = {
                "1m": "1min",
                "5m": "5min",
                "15m": "15min",
                "30m": "30min",
                "1h": "1h",
                "1d": "1d",
            }
            interval = interval_map.get(timeframe, "1min")

            df = self.get_historical_quotes(contract, start_date, end_date, interval)

            if df.empty:
                return []

            bars = []
            for _, row in df.tail(limit).iterrows():
                bars.append(Bar(
                    ts=int(row.get("timestamp", datetime.now()).timestamp() * 1000),
                    symbol=symbol,
                    open=Decimal(str(row.get("open", 0))),
                    high=Decimal(str(row.get("high", 0))),
                    low=Decimal(str(row.get("low", 0))),
                    close=Decimal(str(row.get("close", 0))),
                    volume=Decimal(str(row.get("volume", 0))),
                ))
            return bars

        else:
            # This is a stock - use stock data endpoint
            self._ensure_rate_limit()

            try:
                client = self._get_client()

                end_date = end.date() if end else date.today()
                start_date = start.date() if start else end_date - timedelta(days=30)

                data = client.get_hist_stock(
                    root=symbol.upper(),
                    start_date=start_date.strftime("%Y%m%d"),
                    end_date=end_date.strftime("%Y%m%d"),
                )

                if data is None or data.empty:
                    return []

                bars = []
                for _, row in data.tail(limit).iterrows():
                    bars.append(Bar(
                        ts=int(datetime.combine(row.get("date", date.today()), datetime.min.time()).timestamp() * 1000),
                        symbol=symbol.upper(),
                        open=Decimal(str(row.get("open", 0))),
                        high=Decimal(str(row.get("high", 0))),
                        low=Decimal(str(row.get("low", 0))),
                        close=Decimal(str(row.get("close", 0))),
                        volume=Decimal(str(row.get("volume", 0))),
                    ))
                return bars

            except Exception as e:
                logger.error(f"Failed to get bars for {symbol}: {e}")
                return []

    def get_tick(self, symbol: str):
        """Get current tick - delegate to get_underlying_price or get_quote."""
        if len(symbol) == 21 and symbol[12] in ("C", "P"):
            quote = self.get_quote(self._occ_to_contract(symbol))
            if quote:
                return quote
        else:
            price = self.get_underlying_price(symbol)
            if price:
                return {"price": price, "symbol": symbol}
        return None

    def close(self) -> None:
        """Close adapter and cleanup resources."""
        self._client = None
        self._chain_cache.clear()
        logger.info("Theta Data adapter closed")


# =============================================================================
# Factory Functions
# =============================================================================

def create_theta_data_adapter(
    username: Optional[str] = None,
    password: Optional[str] = None,
    api_key: Optional[str] = None,
    use_terminal: bool = False,
    rate_limit_per_min: int = DEFAULT_RATE_LIMIT_PER_MIN,
) -> ThetaDataOptionsAdapter:
    """
    Create Theta Data options adapter.

    Args:
        username: Theta Data username
        password: Theta Data password
        api_key: API key (optional)
        use_terminal: Use terminal API
        rate_limit_per_min: Rate limit

    Returns:
        ThetaDataOptionsAdapter instance
    """
    config = ThetaDataConfig(
        username=username,
        password=password,
        api_key=api_key,
        use_terminal=use_terminal,
        rate_limit_per_min=rate_limit_per_min,
    )

    return ThetaDataOptionsAdapter(config=config)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constants
    "THETA_DATA_AVAILABLE",
    "THETA_DATA_BASE_URL",
    # Config
    "ThetaDataConfig",
    # Data classes
    "ThetaDataQuote",
    "ThetaDataTrade",
    "ThetaDataChain",
    # Adapter
    "ThetaDataOptionsAdapter",
    # Factory
    "create_theta_data_adapter",
]
