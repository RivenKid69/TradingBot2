# -*- coding: utf-8 -*-
"""
adapters/polygon/options.py
Polygon.io Options Historical Data Adapter.

This module provides historical options data from Polygon.io:
- Historical EOD chain snapshots (2018+)
- Historical NBBO quotes
- Options aggregates (OHLCV)
- Corporate actions impact on options

API Reference:
    https://polygon.io/docs/options/getting-started

Data Coverage:
    - US listed equity options (OPRA feed)
    - Historical data back to 2018
    - EOD snapshots and intraday quotes

Rate Limits:
    - Basic: 5 calls/minute
    - Starter: unlimited REST
    - Developer+: unlimited REST, real-time

OCC Symbology:
    Format: SYMBOL(6) + YYMMDD + C/P + STRIKE(8)
    Example: "O:AAPL241220C00200000" (Polygon format with O: prefix)

Usage:
    adapter = PolygonOptionsAdapter(config={"api_key": "..."})

    # Get historical chain snapshot
    chain = adapter.get_historical_chain("AAPL", date(2024, 1, 15))

    # Get historical quotes for specific contract
    quotes = adapter.get_historical_quotes(
        "O:AAPL241220C00200000",
        date(2024, 1, 1),
        date(2024, 1, 31),
    )

Phase 2: US Exchange Adapters
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from core_options import OptionsContractSpec, OptionType

from .market_data import PolygonMarketDataAdapter
from ..models import ExchangeVendor

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class PolygonOptionsContract:
    """
    Polygon options contract representation.

    Attributes:
        ticker: Polygon options ticker (e.g., "O:AAPL241220C00200000")
        underlying: Underlying symbol
        expiration: Expiration date
        strike: Strike price
        option_type: CALL or PUT
        contract_type: Contract style (american/european)
        shares_per_contract: Usually 100
        cfi: CFI code
        primary_exchange: Primary exchange
    """

    ticker: str
    underlying: str
    expiration: date
    strike: Decimal
    option_type: OptionType
    contract_type: str = "american"
    shares_per_contract: int = 100
    cfi: Optional[str] = None
    primary_exchange: Optional[str] = None

    def to_occ_symbol(self) -> str:
        """Convert to standard OCC symbol (without O: prefix)."""
        # Pad underlying to 6 chars
        underlying_padded = self.underlying.upper().ljust(6)

        # Format date
        date_str = self.expiration.strftime("%y%m%d")

        # Option type
        opt_char = "C" if self.option_type == OptionType.CALL else "P"

        # Strike: 8 digits, multiply by 1000 (3 decimal places implied)
        strike_int = int(self.strike * 1000)
        strike_str = str(strike_int).zfill(8)

        return f"{underlying_padded}{date_str}{opt_char}{strike_str}"

    def to_contract_spec(self) -> OptionsContractSpec:
        """
        Convert to OptionsContractSpec.

        Returns:
            OptionsContractSpec with all required fields populated.
            symbol is set to standard OCC format (e.g., "AAPL  241220C00200000").
        """
        return OptionsContractSpec(
            symbol=self.to_occ_symbol(),
            underlying=self.underlying,
            strike=self.strike,
            expiration=self.expiration,
            option_type=self.option_type,
            multiplier=self.shares_per_contract,
            exercise_style="american" if self.contract_type == "american" else "european",
        )


@dataclass
class PolygonOptionsQuote:
    """
    Polygon options quote data.

    Attributes:
        contract: Options contract
        timestamp: Quote timestamp
        bid: Bid price
        ask: Ask price
        bid_size: Bid size in contracts
        ask_size: Ask size in contracts
        last_price: Last trade price
        volume: Trading volume
        open_interest: Open interest
        underlying_price: Underlying price at quote time
    """

    contract: PolygonOptionsContract
    timestamp: datetime
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    last_price: Optional[Decimal] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    underlying_price: Optional[Decimal] = None

    # Greeks if available
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    iv: Optional[float] = None


@dataclass
class PolygonOptionsSnapshot:
    """
    Polygon options chain snapshot.

    Attributes:
        underlying: Underlying symbol
        snapshot_date: Date of snapshot
        underlying_price: Underlying price
        quotes: List of option quotes
    """

    underlying: str
    snapshot_date: date
    underlying_price: Optional[Decimal] = None
    quotes: List[PolygonOptionsQuote] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert snapshot to DataFrame."""
        records = []
        for q in self.quotes:
            records.append(
                {
                    "ticker": q.contract.ticker,
                    "underlying": q.contract.underlying,
                    "expiration": q.contract.expiration,
                    "strike": float(q.contract.strike),
                    "option_type": q.contract.option_type.value,
                    "bid": float(q.bid) if q.bid else None,
                    "ask": float(q.ask) if q.ask else None,
                    "bid_size": q.bid_size,
                    "ask_size": q.ask_size,
                    "last_price": float(q.last_price) if q.last_price else None,
                    "volume": q.volume,
                    "open_interest": q.open_interest,
                    "underlying_price": float(q.underlying_price) if q.underlying_price else None,
                    "iv": q.iv,
                    "delta": q.delta,
                    "gamma": q.gamma,
                    "theta": q.theta,
                    "vega": q.vega,
                    "timestamp": q.timestamp,
                }
            )

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values(["expiration", "strike", "option_type"])
        return df


# =============================================================================
# OCC SYMBOL UTILITIES
# =============================================================================


def polygon_ticker_to_occ(polygon_ticker: str) -> str:
    """
    Convert Polygon options ticker to OCC symbol.

    Args:
        polygon_ticker: Polygon format (e.g., "O:AAPL241220C00200000")

    Returns:
        OCC symbol (e.g., "AAPL  241220C00200000")
    """
    # Remove O: prefix if present
    ticker = polygon_ticker.lstrip("O:")

    # Parse components using regex
    # Format: SYMBOL + YYMMDD + C/P + STRIKE(8)
    match = re.match(r"^([A-Z]+)(\d{6})([CP])(\d{8})$", ticker)
    if not match:
        raise ValueError(f"Invalid Polygon options ticker: {polygon_ticker}")

    symbol, date_str, opt_type, strike_str = match.groups()

    # Pad symbol to 6 chars
    symbol_padded = symbol.ljust(6)

    return f"{symbol_padded}{date_str}{opt_type}{strike_str}"


def occ_to_polygon_ticker(occ_symbol: str) -> str:
    """
    Convert OCC symbol to Polygon options ticker.

    Args:
        occ_symbol: OCC format (e.g., "AAPL  241220C00200000")

    Returns:
        Polygon ticker (e.g., "O:AAPL241220C00200000")
    """
    # Remove spaces
    clean = occ_symbol.replace(" ", "")

    # Parse components
    match = re.match(r"^([A-Z]+)(\d{6})([CP])(\d{8})$", clean)
    if not match:
        raise ValueError(f"Invalid OCC symbol: {occ_symbol}")

    symbol, date_str, opt_type, strike_str = match.groups()

    return f"O:{symbol}{date_str}{opt_type}{strike_str}"


def parse_polygon_ticker(polygon_ticker: str) -> Tuple[str, date, OptionType, Decimal]:
    """
    Parse Polygon options ticker into components.

    Args:
        polygon_ticker: Polygon format (e.g., "O:AAPL241220C00200000")

    Returns:
        Tuple of (underlying, expiration, option_type, strike)
    """
    # Remove O: prefix if present
    ticker = polygon_ticker.lstrip("O:")

    # Parse components
    match = re.match(r"^([A-Z]+)(\d{6})([CP])(\d{8})$", ticker)
    if not match:
        raise ValueError(f"Invalid Polygon options ticker: {polygon_ticker}")

    symbol, date_str, opt_char, strike_str = match.groups()

    # Parse date (YYMMDD)
    year = 2000 + int(date_str[:2])
    month = int(date_str[2:4])
    day = int(date_str[4:6])
    expiration = date(year, month, day)

    # Option type
    option_type = OptionType.CALL if opt_char == "C" else OptionType.PUT

    # Strike (8 digits, 3 decimal places implied)
    strike = Decimal(strike_str) / Decimal("1000")

    return symbol, expiration, option_type, strike


# =============================================================================
# POLYGON OPTIONS ADAPTER
# =============================================================================


class PolygonOptionsAdapter(PolygonMarketDataAdapter):
    """
    Polygon.io Options Historical Data Adapter.

    Extends PolygonMarketDataAdapter to provide options-specific data:
    - Historical EOD chain snapshots
    - Historical NBBO quotes
    - Options aggregates (OHLCV)
    - Contract reference data

    Configuration:
        api_key: Polygon.io API key (or env POLYGON_API_KEY)
        timeout: Request timeout in seconds (default: 30)
        retries: Number of retry attempts (default: 3)
        cache_chains: Enable chain caching (default: True)
        cache_ttl_sec: Cache TTL in seconds (default: 300)

    Example:
        adapter = PolygonOptionsAdapter(config={
            "api_key": "your_api_key",
            "cache_chains": True,
        })

        # Get EOD chain snapshot
        chain_df = adapter.get_historical_chain("AAPL", date(2024, 1, 15))

        # Get historical quotes
        quotes_df = adapter.get_historical_quotes(
            "O:AAPL241220C00200000",
            date(2024, 1, 1),
            date(2024, 1, 31),
        )
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.POLYGON,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor=vendor, config=config)

        # Options-specific config
        self._cache_chains = bool(self._config.get("cache_chains", True))
        self._cache_ttl_sec = float(self._config.get("cache_ttl_sec", 300.0))

        # Chain cache: (underlying, date) -> (snapshot, cached_at)
        self._chain_cache: Dict[Tuple[str, date], Tuple[PolygonOptionsSnapshot, float]] = {}

        # Rate limiting
        self._last_request_time: float = 0.0
        self._min_request_interval: float = 0.1  # 100ms between requests

    def _ensure_rate_limit(self) -> None:
        """Ensure minimum interval between requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _get_cache_key(self, underlying: str, snapshot_date: date) -> Tuple[str, date]:
        """Generate cache key for chain snapshot."""
        return (underlying.upper(), snapshot_date)

    def _get_cached_chain(
        self,
        underlying: str,
        snapshot_date: date,
    ) -> Optional[PolygonOptionsSnapshot]:
        """Get cached chain snapshot if valid."""
        if not self._cache_chains:
            return None

        key = self._get_cache_key(underlying, snapshot_date)
        if key not in self._chain_cache:
            return None

        snapshot, cached_at = self._chain_cache[key]
        if time.time() - cached_at > self._cache_ttl_sec:
            del self._chain_cache[key]
            return None

        return snapshot

    def _cache_chain(
        self,
        underlying: str,
        snapshot_date: date,
        snapshot: PolygonOptionsSnapshot,
    ) -> None:
        """Cache chain snapshot."""
        if self._cache_chains:
            key = self._get_cache_key(underlying, snapshot_date)
            self._chain_cache[key] = (snapshot, time.time())

    # -------------------------------------------------------------------------
    # Contract Reference
    # -------------------------------------------------------------------------

    def get_contracts(
        self,
        underlying: str,
        expiration_date: Optional[date] = None,
        expiration_date_gte: Optional[date] = None,
        expiration_date_lte: Optional[date] = None,
        contract_type: Optional[str] = None,  # "call" or "put"
        strike_price: Optional[float] = None,
        strike_price_gte: Optional[float] = None,
        strike_price_lte: Optional[float] = None,
        limit: int = 1000,
    ) -> List[PolygonOptionsContract]:
        """
        Get options contracts matching criteria.

        Args:
            underlying: Underlying symbol
            expiration_date: Exact expiration date
            expiration_date_gte: Expiration >= this date
            expiration_date_lte: Expiration <= this date
            contract_type: "call" or "put"
            strike_price: Exact strike price
            strike_price_gte: Strike >= this price
            strike_price_lte: Strike <= this price
            limit: Maximum contracts to return

        Returns:
            List of matching contracts
        """
        client = self._get_rest_client()
        self._ensure_rate_limit()

        contracts: List[PolygonOptionsContract] = []

        try:
            # Build params
            params: Dict[str, Any] = {
                "underlying_ticker": underlying.upper(),
                "limit": min(limit, 1000),
            }

            if expiration_date:
                params["expiration_date"] = expiration_date.isoformat()
            if expiration_date_gte:
                params["expiration_date.gte"] = expiration_date_gte.isoformat()
            if expiration_date_lte:
                params["expiration_date.lte"] = expiration_date_lte.isoformat()
            if contract_type:
                params["contract_type"] = contract_type.lower()
            if strike_price is not None:
                params["strike_price"] = strike_price
            if strike_price_gte is not None:
                params["strike_price.gte"] = strike_price_gte
            if strike_price_lte is not None:
                params["strike_price.lte"] = strike_price_lte

            # Fetch contracts
            results = client.list_options_contracts(**params)

            for item in results:
                try:
                    contract = PolygonOptionsContract(
                        ticker=item.ticker,
                        underlying=item.underlying_ticker,
                        expiration=(
                            datetime.strptime(item.expiration_date, "%Y-%m-%d").date()
                            if isinstance(item.expiration_date, str)
                            else item.expiration_date
                        ),
                        strike=Decimal(str(item.strike_price)),
                        option_type=(
                            OptionType.CALL if item.contract_type == "call" else OptionType.PUT
                        ),
                        contract_type=item.exercise_style or "american",
                        shares_per_contract=item.shares_per_contract or 100,
                        cfi=getattr(item, "cfi", None),
                        primary_exchange=getattr(item, "primary_exchange", None),
                    )
                    contracts.append(contract)
                except Exception as e:
                    logger.warning(
                        f"Failed to parse contract {getattr(item, 'ticker', 'unknown')}: {e}"
                    )

            logger.debug(f"Found {len(contracts)} contracts for {underlying}")
            return contracts

        except Exception as e:
            logger.error(f"Failed to get contracts for {underlying}: {e}")
            return []

    def get_expirations(
        self,
        underlying: str,
        min_date: Optional[date] = None,
        max_date: Optional[date] = None,
    ) -> List[date]:
        """
        Get available expiration dates for underlying.

        Args:
            underlying: Underlying symbol
            min_date: Minimum expiration date
            max_date: Maximum expiration date

        Returns:
            List of expiration dates, sorted ascending
        """
        # Get contracts and extract unique expirations
        contracts = self.get_contracts(
            underlying=underlying,
            expiration_date_gte=min_date or date.today(),
            expiration_date_lte=max_date,
            limit=1000,
        )

        expirations = sorted(set(c.expiration for c in contracts))
        return expirations

    def get_strikes(
        self,
        underlying: str,
        expiration: date,
    ) -> List[Decimal]:
        """
        Get available strikes for underlying/expiration.

        Args:
            underlying: Underlying symbol
            expiration: Expiration date

        Returns:
            List of strikes, sorted ascending
        """
        contracts = self.get_contracts(
            underlying=underlying,
            expiration_date=expiration,
        )

        strikes = sorted(set(c.strike for c in contracts))
        return strikes

    # -------------------------------------------------------------------------
    # Historical Chain Snapshots
    # -------------------------------------------------------------------------

    def get_historical_chain(
        self,
        underlying: str,
        snapshot_date: date,
        expiration: Optional[date] = None,
        strike_range_pct: float = 0.3,
    ) -> pd.DataFrame:
        """
        Get historical EOD chain snapshot.

        Retrieves the options chain as it was at market close on the given date.

        Args:
            underlying: Underlying symbol (e.g., "AAPL")
            snapshot_date: Date for snapshot (EOD)
            expiration: Specific expiration to filter (optional)
            strike_range_pct: Strike range as % of underlying price (default 30%)

        Returns:
            DataFrame with columns:
            - ticker: Polygon options ticker
            - underlying: Underlying symbol
            - expiration: Expiration date
            - strike: Strike price
            - option_type: "call" or "put"
            - bid, ask: Quote prices
            - last_price: Last trade price
            - volume: Trading volume
            - open_interest: Open interest
            - iv: Implied volatility
            - delta, gamma, theta, vega: Greeks
            - underlying_price: Underlying price
            - timestamp: Quote timestamp
        """
        # Check cache
        cached = self._get_cached_chain(underlying, snapshot_date)
        if cached:
            df = cached.to_dataframe()
            if expiration:
                df = df[df["expiration"] == expiration]
            return df

        client = self._get_rest_client()
        self._ensure_rate_limit()

        # Get underlying price on snapshot date
        underlying_price = self._get_underlying_price_on_date(underlying, snapshot_date)

        # Determine expiration filter
        if expiration:
            exp_filter = expiration
        else:
            # Get nearest expirations (within 60 days)
            exp_filter = None

        # Get contracts
        contracts = self.get_contracts(
            underlying=underlying,
            expiration_date=exp_filter,
            expiration_date_gte=snapshot_date if not exp_filter else None,
            expiration_date_lte=(snapshot_date + timedelta(days=60)) if not exp_filter else None,
        )

        if not contracts:
            logger.warning(f"No contracts found for {underlying} on {snapshot_date}")
            return pd.DataFrame()

        # Filter by strike range if underlying price available
        if underlying_price and strike_range_pct > 0:
            min_strike = underlying_price * Decimal(str(1 - strike_range_pct))
            max_strike = underlying_price * Decimal(str(1 + strike_range_pct))
            contracts = [c for c in contracts if min_strike <= c.strike <= max_strike]

        # Fetch snapshots for contracts
        quotes: List[PolygonOptionsQuote] = []

        # Process in batches to respect rate limits
        batch_size = 50
        for i in range(0, len(contracts), batch_size):
            batch = contracts[i : i + batch_size]

            for contract in batch:
                try:
                    self._ensure_rate_limit()
                    quote = self._get_contract_snapshot(contract, snapshot_date, underlying_price)
                    if quote:
                        quotes.append(quote)
                except Exception as e:
                    logger.debug(f"Failed to get snapshot for {contract.ticker}: {e}")

        # Create snapshot object
        snapshot = PolygonOptionsSnapshot(
            underlying=underlying.upper(),
            snapshot_date=snapshot_date,
            underlying_price=underlying_price,
            quotes=quotes,
        )

        # Cache result
        self._cache_chain(underlying, snapshot_date, snapshot)

        return snapshot.to_dataframe()

    def _get_underlying_price_on_date(
        self,
        underlying: str,
        snapshot_date: date,
    ) -> Optional[Decimal]:
        """Get underlying closing price on given date."""
        client = self._get_rest_client()

        try:
            # Get daily bar
            date_str = snapshot_date.isoformat()
            aggs = client.get_aggs(
                ticker=underlying.upper(),
                multiplier=1,
                timespan="day",
                from_=date_str,
                to=date_str,
                limit=1,
            )

            for agg in aggs:
                return Decimal(str(agg.close))

        except Exception as e:
            logger.warning(
                f"Failed to get underlying price for {underlying} on {snapshot_date}: {e}"
            )

        return None

    def _get_contract_snapshot(
        self,
        contract: PolygonOptionsContract,
        snapshot_date: date,
        underlying_price: Optional[Decimal],
    ) -> Optional[PolygonOptionsQuote]:
        """Get snapshot for a single contract on given date."""
        client = self._get_rest_client()

        try:
            # Try to get daily aggregates for the contract
            date_str = snapshot_date.isoformat()
            aggs = client.get_aggs(
                ticker=contract.ticker,
                multiplier=1,
                timespan="day",
                from_=date_str,
                to=date_str,
                limit=1,
            )

            for agg in aggs:
                # Create quote from aggregate data
                return PolygonOptionsQuote(
                    contract=contract,
                    timestamp=datetime.combine(snapshot_date, datetime.max.time()),
                    bid=None,  # Not available in aggregates
                    ask=None,
                    last_price=Decimal(str(agg.close)) if agg.close else None,
                    volume=int(agg.volume) if agg.volume else None,
                    open_interest=getattr(agg, "open_interest", None),
                    underlying_price=underlying_price,
                )

        except Exception as e:
            logger.debug(f"No data for {contract.ticker} on {snapshot_date}: {e}")

        return None

    # -------------------------------------------------------------------------
    # Historical Quotes
    # -------------------------------------------------------------------------

    def get_historical_quotes(
        self,
        contract_symbol: str,
        start_date: date,
        end_date: date,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """
        Get historical NBBO quotes/aggregates for an options contract.

        Args:
            contract_symbol: Polygon ticker (e.g., "O:AAPL241220C00200000")
                            or OCC symbol (e.g., "AAPL  241220C00200000")
            start_date: Start date
            end_date: End date
            timeframe: Bar timeframe ("1m", "5m", "1h", "1d")

        Returns:
            DataFrame with columns:
            - timestamp: Bar timestamp
            - open, high, low, close: OHLC prices
            - volume: Trading volume
            - vwap: Volume-weighted average price
            - num_trades: Number of trades
        """
        # Normalize to Polygon format
        if not contract_symbol.startswith("O:"):
            contract_symbol = occ_to_polygon_ticker(contract_symbol)

        client = self._get_rest_client()
        self._ensure_rate_limit()

        # Parse timeframe
        from .market_data import _timeframe_to_polygon

        multiplier, timespan = _timeframe_to_polygon(timeframe)

        try:
            aggs = client.get_aggs(
                ticker=contract_symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=start_date.isoformat(),
                to=end_date.isoformat(),
                limit=50000,
                sort="asc",
            )

            records = []
            for agg in aggs:
                ts = datetime.fromtimestamp(agg.timestamp / 1000, tz=timezone.utc)
                records.append(
                    {
                        "timestamp": ts,
                        "open": float(agg.open) if agg.open else None,
                        "high": float(agg.high) if agg.high else None,
                        "low": float(agg.low) if agg.low else None,
                        "close": float(agg.close) if agg.close else None,
                        "volume": int(agg.volume) if agg.volume else 0,
                        "vwap": float(agg.vwap) if hasattr(agg, "vwap") and agg.vwap else None,
                        "num_trades": (
                            int(agg.transactions) if hasattr(agg, "transactions") else None
                        ),
                    }
                )

            df = pd.DataFrame(records)
            if not df.empty:
                df = df.sort_values("timestamp")

            logger.debug(f"Fetched {len(df)} bars for {contract_symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to get historical quotes for {contract_symbol}: {e}")
            return pd.DataFrame()

    def get_historical_trades(
        self,
        contract_symbol: str,
        trade_date: date,
        limit: int = 50000,
    ) -> pd.DataFrame:
        """
        Get historical trades for an options contract.

        Args:
            contract_symbol: Polygon ticker or OCC symbol
            trade_date: Date to get trades for
            limit: Maximum trades to return

        Returns:
            DataFrame with columns:
            - timestamp: Trade timestamp
            - price: Trade price
            - size: Trade size in contracts
            - exchange: Exchange code
            - conditions: Trade condition codes
        """
        # Normalize to Polygon format
        if not contract_symbol.startswith("O:"):
            contract_symbol = occ_to_polygon_ticker(contract_symbol)

        client = self._get_rest_client()
        self._ensure_rate_limit()

        try:
            # Get trades for the date
            trades = client.list_trades(
                ticker=contract_symbol,
                timestamp_gte=f"{trade_date.isoformat()}T00:00:00Z",
                timestamp_lte=f"{trade_date.isoformat()}T23:59:59Z",
                limit=min(limit, 50000),
                sort="timestamp",
            )

            records = []
            for trade in trades:
                ts = datetime.fromtimestamp(
                    trade.sip_timestamp / 1_000_000_000, tz=timezone.utc  # Nanoseconds to seconds
                )
                records.append(
                    {
                        "timestamp": ts,
                        "price": float(trade.price) if trade.price else None,
                        "size": int(trade.size) if trade.size else None,
                        "exchange": trade.exchange,
                        "conditions": trade.conditions,
                    }
                )

            df = pd.DataFrame(records)
            if not df.empty:
                df = df.sort_values("timestamp")

            logger.debug(f"Fetched {len(df)} trades for {contract_symbol} on {trade_date}")
            return df

        except Exception as e:
            logger.error(f"Failed to get trades for {contract_symbol}: {e}")
            return pd.DataFrame()

    # -------------------------------------------------------------------------
    # Options Greeks (from Snapshot API)
    # -------------------------------------------------------------------------

    def get_current_snapshot(
        self,
        underlying: str,
        contract_symbol: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get current options snapshot with Greeks.

        Note: Requires real-time subscription for live data.

        Args:
            underlying: Underlying symbol
            contract_symbol: Specific contract (optional)

        Returns:
            Snapshot data with Greeks if available
        """
        client = self._get_rest_client()
        self._ensure_rate_limit()

        try:
            if contract_symbol:
                # Get single contract snapshot
                if not contract_symbol.startswith("O:"):
                    contract_symbol = occ_to_polygon_ticker(contract_symbol)

                # Remove O: prefix for API call
                ticker_clean = contract_symbol.lstrip("O:")

                snapshot = client.get_snapshot_option(
                    underlying.upper(),
                    ticker_clean,
                )

                return self._parse_snapshot_response(snapshot)
            else:
                # Get full chain snapshot
                snapshots = client.list_snapshot_options_chain(
                    underlying.upper(),
                )

                results = []
                for snap in snapshots:
                    parsed = self._parse_snapshot_response(snap)
                    if parsed:
                        results.append(parsed)

                return {"chain": results}

        except Exception as e:
            logger.error(f"Failed to get snapshot for {underlying}: {e}")
            return None

    def _parse_snapshot_response(self, snap: Any) -> Optional[Dict[str, Any]]:
        """Parse Polygon snapshot response."""
        if not snap:
            return None

        try:
            details = getattr(snap, "details", {}) or {}
            greeks = getattr(snap, "greeks", {}) or {}
            underlying = getattr(snap, "underlying_asset", {}) or {}
            day = getattr(snap, "day", {}) or {}
            last_quote = getattr(snap, "last_quote", {}) or {}

            return {
                "ticker": getattr(details, "ticker", None),
                "strike": getattr(details, "strike_price", None),
                "expiration": getattr(details, "expiration_date", None),
                "contract_type": getattr(details, "contract_type", None),
                # Greeks
                "iv": getattr(greeks, "implied_volatility", None),
                "delta": getattr(greeks, "delta", None),
                "gamma": getattr(greeks, "gamma", None),
                "theta": getattr(greeks, "theta", None),
                "vega": getattr(greeks, "vega", None),
                # Underlying
                "underlying_price": getattr(underlying, "price", None),
                # Day stats
                "volume": getattr(day, "volume", None),
                "open_interest": getattr(day, "open_interest", None),
                "close": getattr(day, "close", None),
                # Quote
                "bid": getattr(last_quote, "bid", None),
                "ask": getattr(last_quote, "ask", None),
                "bid_size": getattr(last_quote, "bid_size", None),
                "ask_size": getattr(last_quote, "ask_size", None),
            }

        except Exception as e:
            logger.debug(f"Failed to parse snapshot: {e}")
            return None

    # -------------------------------------------------------------------------
    # Corporate Actions
    # -------------------------------------------------------------------------

    def get_stock_splits(
        self,
        underlying: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Get stock split history that affects options.

        Args:
            underlying: Underlying symbol
            start_date: Start date for splits
            end_date: End date for splits

        Returns:
            DataFrame with split details
        """
        client = self._get_rest_client()
        self._ensure_rate_limit()

        try:
            splits = client.list_splits(
                ticker=underlying.upper(),
                execution_date_gte=start_date.isoformat() if start_date else None,
                execution_date_lte=end_date.isoformat() if end_date else None,
            )

            records = []
            for split in splits:
                records.append(
                    {
                        "ticker": split.ticker,
                        "execution_date": split.execution_date,
                        "split_from": split.split_from,
                        "split_to": split.split_to,
                    }
                )

            return pd.DataFrame(records)

        except Exception as e:
            logger.error(f"Failed to get splits for {underlying}: {e}")
            return pd.DataFrame()

    def get_dividends(
        self,
        underlying: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Get dividend history that affects options pricing.

        Args:
            underlying: Underlying symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with dividend details
        """
        client = self._get_rest_client()
        self._ensure_rate_limit()

        try:
            dividends = client.list_dividends(
                ticker=underlying.upper(),
                ex_dividend_date_gte=start_date.isoformat() if start_date else None,
                ex_dividend_date_lte=end_date.isoformat() if end_date else None,
            )

            records = []
            for div in dividends:
                records.append(
                    {
                        "ticker": div.ticker,
                        "ex_dividend_date": div.ex_dividend_date,
                        "pay_date": getattr(div, "pay_date", None),
                        "record_date": getattr(div, "record_date", None),
                        "cash_amount": getattr(div, "cash_amount", None),
                        "dividend_type": getattr(div, "dividend_type", None),
                    }
                )

            return pd.DataFrame(records)

        except Exception as e:
            logger.error(f"Failed to get dividends for {underlying}: {e}")
            return pd.DataFrame()

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._chain_cache.clear()
        logger.debug("Options chain cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "chain_cache_size": len(self._chain_cache),
            "cache_enabled": self._cache_chains,
            "cache_ttl_sec": self._cache_ttl_sec,
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_polygon_options_adapter(
    api_key: Optional[str] = None,
    cache_chains: bool = True,
    cache_ttl_sec: float = 300.0,
    **kwargs: Any,
) -> PolygonOptionsAdapter:
    """
    Create Polygon options adapter with configuration.

    Args:
        api_key: Polygon API key (default from env)
        cache_chains: Enable chain caching
        cache_ttl_sec: Cache TTL in seconds
        **kwargs: Additional config options

    Returns:
        Configured PolygonOptionsAdapter

    Example:
        adapter = create_polygon_options_adapter(
            api_key="your_key",
            cache_chains=True,
        )
    """
    config = {
        "api_key": api_key or os.environ.get("POLYGON_API_KEY"),
        "cache_chains": cache_chains,
        "cache_ttl_sec": cache_ttl_sec,
        **kwargs,
    }

    return PolygonOptionsAdapter(config=config)
