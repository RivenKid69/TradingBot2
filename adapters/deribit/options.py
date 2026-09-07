# -*- coding: utf-8 -*-
"""
adapters/deribit/options.py

Deribit Crypto Options Market Data and Order Execution Adapters.

This module implements adapters for Deribit, the leading crypto options exchange.
Key features:
    - Inverse settlement (P&L in underlying crypto, not USD)
    - European exercise only (no early exercise)
    - 24/7 trading with varying liquidity
    - DVOL index integration

Deribit Instrument Naming Convention:
    BTC-28MAR25-100000-C
    ^   ^       ^      ^
    |   |       |      └── Option type: C (call) or P (put)
    |   |       └── Strike price in USD
    |   └── Expiration date: DDMMMYY (e.g., 28MAR25 = March 28, 2025)
    └── Underlying: BTC or ETH

Strike Conventions:
    - BTC: $1000 increments (e.g., 50000, 51000, 52000)
    - ETH: $50 increments (e.g., 3000, 3050, 3100)

Expiration Pattern (all at 08:00 UTC):
    - Daily: Every day
    - Weekly: Every Friday
    - Monthly: Last Friday of month
    - Quarterly: Last Friday of Mar/Jun/Sep/Dec

IMPLEMENTATION STATUS
=====================
This adapter is **REST-only**. Streaming methods raise NotImplementedError.

- REST Market Data: ✅ Implemented (get_option_chain, get_option_quote, etc.)
- REST Order Execution: ✅ Implemented (place_order, cancel_order, etc.)
- WebSocket Streaming: ❌ Not implemented in this adapter

For real-time streaming, use `DeribitWebSocketClient` from `adapters/deribit/websocket.py`.

The REST adapter is suitable for:
- Historical data analysis
- Position queries
- Order management with polling

For low-latency real-time data, WebSocket is recommended.

Tech Debt Tracking: docs/reports/TECH_DEBT_REGISTRY.md#adapter-deribit-rest-only

References:
    - Deribit API: https://docs.deribit.com/
    - DVOL Methodology: https://www.deribit.com/pages/docs/volatility-index
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import requests

from adapters.base import MarketDataAdapter, OrderExecutionAdapter, OrderResult
from adapters.models import AccountInfo, ExchangeVendor, MarketType
from core_models import (
    Bar,
    Tick,
    Order as CoreOrder,
    ExecReport,
    Position as CorePosition,
    Side as CoreSide,
    OrderType as CoreOrderType,
    ExecStatus,
    Liquidity,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

DERIBIT_API_URL = "https://www.deribit.com/api/v2"
DERIBIT_TESTNET_URL = "https://test.deribit.com/api/v2"

# Strike increments by underlying
STRIKE_INCREMENTS = {
    "BTC": Decimal("1000"),
    "ETH": Decimal("50"),
}

# Minimum order sizes
MIN_ORDER_SIZE = {
    "BTC": Decimal("0.1"),  # 0.1 BTC contracts
    "ETH": Decimal("1.0"),  # 1.0 ETH contracts
}

# Tick sizes for options (in underlying)
TICK_SIZE = {
    "BTC": Decimal("0.0001"),  # 0.0001 BTC
    "ETH": Decimal("0.0005"),  # 0.0005 ETH
}

# Month code mapping for instrument names
MONTH_CODES = {
    1: "JAN",
    2: "FEB",
    3: "MAR",
    4: "APR",
    5: "MAY",
    6: "JUN",
    7: "JUL",
    8: "AUG",
    9: "SEP",
    10: "OCT",
    11: "NOV",
    12: "DEC",
}
MONTH_CODES_REVERSE = {v: k for k, v in MONTH_CODES.items()}


# =============================================================================
# Enums
# =============================================================================


class DeribitOptionType(str, Enum):
    """Option type for Deribit options."""

    CALL = "call"
    PUT = "put"

    @classmethod
    def from_string(cls, s: str) -> "DeribitOptionType":
        """Parse option type from string."""
        s = s.upper().strip()
        if s in ("C", "CALL"):
            return cls.CALL
        elif s in ("P", "PUT"):
            return cls.PUT
        raise ValueError(f"Invalid option type: {s}")


class DeribitOrderType(str, Enum):
    """Order types supported by Deribit."""

    LIMIT = "limit"
    MARKET = "market"
    STOP_LIMIT = "stop_limit"
    STOP_MARKET = "stop_market"


class DeribitOrderState(str, Enum):
    """Order states on Deribit."""

    OPEN = "open"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    UNTRIGGERED = "untriggered"  # For stop orders


class DeribitTimeInForce(str, Enum):
    """Time in force options for Deribit orders."""

    GTC = "good_til_cancelled"
    IOC = "immediate_or_cancel"
    FOK = "fill_or_kill"


class DeribitDirection(str, Enum):
    """Trade direction."""

    BUY = "buy"
    SELL = "sell"


# =============================================================================
# Utility Functions
# =============================================================================


def parse_deribit_instrument_name(instrument_name: str) -> Dict[str, Any]:
    """
    Parse a Deribit instrument name into components.

    Format: UNDERLYING-DDMMMYY-STRIKE-TYPE
    Example: BTC-28MAR25-100000-C

    Args:
        instrument_name: Deribit instrument name (e.g., "BTC-28MAR25-100000-C")

    Returns:
        Dictionary with parsed components:
            - underlying: str (e.g., "BTC")
            - expiration: date
            - strike: Decimal
            - option_type: DeribitOptionType

    Raises:
        ValueError: If instrument name is invalid
    """
    pattern = r"^([A-Z]+)-(\d{1,2})([A-Z]{3})(\d{2})-(\d+)-([CP])$"
    match = re.match(pattern, instrument_name.upper())

    if not match:
        raise ValueError(f"Invalid Deribit instrument name: {instrument_name}")

    underlying = match.group(1)
    day = int(match.group(2))
    month_str = match.group(3)
    year_short = int(match.group(4))
    strike = Decimal(match.group(5))
    option_type_str = match.group(6)

    # Parse month
    if month_str not in MONTH_CODES_REVERSE:
        raise ValueError(f"Invalid month code: {month_str}")
    month = MONTH_CODES_REVERSE[month_str]

    # Parse year (2-digit to 4-digit)
    year = 2000 + year_short if year_short < 50 else 1900 + year_short

    # Create expiration date
    try:
        expiration = date(year, month, day)
    except ValueError as e:
        raise ValueError(f"Invalid expiration date in {instrument_name}: {e}")

    # Parse option type
    option_type = DeribitOptionType.CALL if option_type_str == "C" else DeribitOptionType.PUT

    return {
        "underlying": underlying,
        "expiration": expiration,
        "strike": strike,
        "option_type": option_type,
    }


def create_deribit_instrument_name(
    underlying: str,
    expiration: date,
    strike: Union[Decimal, int, float],
    option_type: Union[DeribitOptionType, str],
) -> str:
    """
    Create a Deribit instrument name from components.

    Args:
        underlying: Underlying asset (e.g., "BTC", "ETH")
        expiration: Expiration date
        strike: Strike price in USD
        option_type: Call or put

    Returns:
        Deribit instrument name (e.g., "BTC-28MAR25-100000-C")
    """
    underlying = underlying.upper()

    # Format strike (no decimals for Deribit)
    strike_int = int(Decimal(str(strike)))

    # Format expiration
    day = expiration.day
    month = MONTH_CODES[expiration.month]
    year = expiration.year % 100

    # Format option type
    if isinstance(option_type, str):
        option_type = DeribitOptionType.from_string(option_type)
    type_char = "C" if option_type == DeribitOptionType.CALL else "P"

    return f"{underlying}-{day}{month}{year:02d}-{strike_int}-{type_char}"


def btc_to_usd(btc_amount: Decimal, btc_price: Decimal) -> Decimal:
    """Convert BTC amount to USD value."""
    return btc_amount * btc_price


def usd_to_btc(usd_amount: Decimal, btc_price: Decimal) -> Decimal:
    """Convert USD amount to BTC value."""
    if btc_price == 0:
        return Decimal("0")
    return usd_amount / btc_price


def eth_to_usd(eth_amount: Decimal, eth_price: Decimal) -> Decimal:
    """Convert ETH amount to USD value."""
    return eth_amount * eth_price


def usd_to_eth(usd_amount: Decimal, eth_price: Decimal) -> Decimal:
    """Convert USD amount to ETH value."""
    if eth_price == 0:
        return Decimal("0")
    return usd_amount / eth_price


def _compute_inverse_call_payoff(
    spot: Decimal,
    strike: Decimal,
) -> Decimal:
    """
    Compute inverse call option payoff (in underlying crypto).

    Formula: max(0, S - K) / S

    This gives the payoff in units of the underlying cryptocurrency,
    not in USD. This is the key difference from standard options.

    Args:
        spot: Spot price at expiration
        strike: Strike price

    Returns:
        Payoff in underlying cryptocurrency units
    """
    if spot <= 0:
        return Decimal("0")
    intrinsic = max(Decimal("0"), spot - strike)
    return intrinsic / spot


def _compute_inverse_put_payoff(
    spot: Decimal,
    strike: Decimal,
) -> Decimal:
    """
    Compute inverse put option payoff (in underlying crypto).

    Formula: max(0, K - S) / S

    Args:
        spot: Spot price at expiration
        strike: Strike price

    Returns:
        Payoff in underlying cryptocurrency units
    """
    if spot <= 0:
        return Decimal("0")
    intrinsic = max(Decimal("0"), strike - spot)
    return intrinsic / spot


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DeribitGreeks:
    """
    Greeks for a Deribit option.

    Note: Deribit provides Greeks in inverse terms (per underlying unit).
    Delta is the most commonly used Greek for crypto options.
    """

    delta: Decimal
    gamma: Decimal
    theta: Decimal  # Per day decay in underlying units
    vega: Decimal  # Sensitivity to 1% IV change
    rho: Decimal  # Interest rate sensitivity

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "DeribitGreeks":
        """Create Greeks from Deribit API response."""
        return cls(
            delta=Decimal(str(data.get("delta", 0))),
            gamma=Decimal(str(data.get("gamma", 0))),
            theta=Decimal(str(data.get("theta", 0))),
            vega=Decimal(str(data.get("vega", 0))),
            rho=Decimal(str(data.get("rho", 0))),
        )


@dataclass
class DeribitOptionContract:
    """
    Represents a Deribit option contract.

    Attributes:
        instrument_name: Full instrument name (e.g., "BTC-28MAR25-100000-C")
        underlying: Base asset (BTC or ETH)
        expiration: Expiration date (08:00 UTC)
        strike: Strike price in USD
        option_type: Call or put
        settlement_currency: Currency for settlement (BTC or ETH)
        contract_size: Size per contract (1.0 for both BTC and ETH options)
        tick_size: Minimum price increment
        min_trade_amount: Minimum order size
        is_active: Whether the contract is currently tradeable
    """

    instrument_name: str
    underlying: str
    expiration: date
    strike: Decimal
    option_type: DeribitOptionType
    settlement_currency: str
    contract_size: Decimal = Decimal("1.0")
    tick_size: Decimal = Decimal("0.0001")
    min_trade_amount: Decimal = Decimal("0.1")
    is_active: bool = True

    @classmethod
    def from_instrument_name(cls, instrument_name: str) -> "DeribitOptionContract":
        """Create contract from instrument name."""
        parsed = parse_deribit_instrument_name(instrument_name)
        underlying = parsed["underlying"]

        return cls(
            instrument_name=instrument_name,
            underlying=underlying,
            expiration=parsed["expiration"],
            strike=parsed["strike"],
            option_type=parsed["option_type"],
            settlement_currency=underlying,
            tick_size=TICK_SIZE.get(underlying, Decimal("0.0001")),
            min_trade_amount=MIN_ORDER_SIZE.get(underlying, Decimal("0.1")),
        )

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "DeribitOptionContract":
        """Create contract from Deribit API response."""
        instrument_name = data["instrument_name"]
        parsed = parse_deribit_instrument_name(instrument_name)

        return cls(
            instrument_name=instrument_name,
            underlying=parsed["underlying"],
            expiration=parsed["expiration"],
            strike=parsed["strike"],
            option_type=parsed["option_type"],
            settlement_currency=data.get("settlement_currency", parsed["underlying"]),
            contract_size=Decimal(str(data.get("contract_size", 1))),
            tick_size=Decimal(str(data.get("tick_size", 0.0001))),
            min_trade_amount=Decimal(str(data.get("min_trade_amount", 0.1))),
            is_active=data.get("is_active", True),
        )

    @property
    def is_call(self) -> bool:
        """Returns True if this is a call option."""
        return self.option_type == DeribitOptionType.CALL

    @property
    def is_put(self) -> bool:
        """Returns True if this is a put option."""
        return self.option_type == DeribitOptionType.PUT

    def days_to_expiry(self, from_date: Optional[date] = None) -> int:
        """Calculate days until expiration."""
        if from_date is None:
            from_date = date.today()
        return (self.expiration - from_date).days


@dataclass
class DeribitOptionQuote:
    """
    Real-time quote for a Deribit option.

    All prices are in the underlying cryptocurrency (inverse pricing).
    """

    instrument_name: str
    timestamp_ms: int
    bid_price: Optional[Decimal]  # Best bid in underlying
    ask_price: Optional[Decimal]  # Best ask in underlying
    bid_size: Optional[Decimal]  # Size at best bid
    ask_size: Optional[Decimal]  # Size at best ask
    last_price: Optional[Decimal]
    mark_price: Decimal  # Mark price for margining
    index_price: Decimal  # Underlying index price (USD)
    underlying_price: Decimal  # Spot price of underlying
    mark_iv: Decimal  # Implied volatility at mark price
    bid_iv: Optional[Decimal]  # IV at bid
    ask_iv: Optional[Decimal]  # IV at ask
    greeks: Optional[DeribitGreeks] = None
    open_interest: Decimal = Decimal("0")
    volume_24h: Decimal = Decimal("0")

    @property
    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price if both bid and ask available."""
        if self.bid_price is not None and self.ask_price is not None:
            return (self.bid_price + self.ask_price) / 2
        return None

    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread."""
        if self.bid_price is not None and self.ask_price is not None:
            return self.ask_price - self.bid_price
        return None

    @property
    def spread_bps(self) -> Optional[Decimal]:
        """Calculate spread in basis points relative to mid."""
        mid = self.mid_price
        spread = self.spread
        if mid is not None and spread is not None and mid > 0:
            return (spread / mid) * Decimal("10000")
        return None

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "DeribitOptionQuote":
        """Create quote from Deribit API response."""
        greeks = None
        if "greeks" in data:
            greeks = DeribitGreeks.from_api_response(data["greeks"])
        elif all(k in data for k in ["delta", "gamma", "theta", "vega", "rho"]):
            greeks = DeribitGreeks(
                delta=Decimal(str(data.get("delta", 0))),
                gamma=Decimal(str(data.get("gamma", 0))),
                theta=Decimal(str(data.get("theta", 0))),
                vega=Decimal(str(data.get("vega", 0))),
                rho=Decimal(str(data.get("rho", 0))),
            )

        return cls(
            instrument_name=data["instrument_name"],
            timestamp_ms=data.get("timestamp", int(time.time() * 1000)),
            bid_price=Decimal(str(data["best_bid_price"])) if data.get("best_bid_price") else None,
            ask_price=Decimal(str(data["best_ask_price"])) if data.get("best_ask_price") else None,
            bid_size=Decimal(str(data["best_bid_amount"])) if data.get("best_bid_amount") else None,
            ask_size=Decimal(str(data["best_ask_amount"])) if data.get("best_ask_amount") else None,
            last_price=Decimal(str(data["last_price"])) if data.get("last_price") else None,
            mark_price=Decimal(str(data.get("mark_price", 0))),
            index_price=Decimal(str(data.get("index_price", 0))),
            underlying_price=Decimal(str(data.get("underlying_price", data.get("index_price", 0)))),
            mark_iv=Decimal(str(data.get("mark_iv", 0))),
            bid_iv=Decimal(str(data["bid_iv"])) if data.get("bid_iv") else None,
            ask_iv=Decimal(str(data["ask_iv"])) if data.get("ask_iv") else None,
            greeks=greeks,
            open_interest=Decimal(str(data.get("open_interest", 0))),
            volume_24h=Decimal(str(data.get("stats", {}).get("volume", 0))),
        )


@dataclass
class DeribitOrderbookLevel:
    """Single level in the order book."""

    price: Decimal  # In underlying
    size: Decimal  # In contracts


@dataclass
class DeribitOrderbook:
    """
    Order book snapshot for a Deribit option.

    Prices are in the underlying cryptocurrency.
    """

    instrument_name: str
    timestamp_ms: int
    bids: List[DeribitOrderbookLevel]  # Sorted by price descending
    asks: List[DeribitOrderbookLevel]  # Sorted by price ascending

    @property
    def best_bid(self) -> Optional[DeribitOrderbookLevel]:
        """Get best bid level."""
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[DeribitOrderbookLevel]:
        """Get best ask level."""
        return self.asks[0] if self.asks else None

    @property
    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / 2
        return None

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "DeribitOrderbook":
        """Create orderbook from Deribit API response."""
        bids = [
            DeribitOrderbookLevel(
                price=Decimal(str(level[0])),
                size=Decimal(str(level[1])),
            )
            for level in data.get("bids", [])
        ]
        asks = [
            DeribitOrderbookLevel(
                price=Decimal(str(level[0])),
                size=Decimal(str(level[1])),
            )
            for level in data.get("asks", [])
        ]

        return cls(
            instrument_name=data["instrument_name"],
            timestamp_ms=data.get("timestamp", int(time.time() * 1000)),
            bids=bids,
            asks=asks,
        )


@dataclass
class DVOLData:
    """
    DVOL (Deribit Volatility Index) data.

    DVOL is Deribit's 30-day constant maturity implied volatility index,
    similar in methodology to the VIX but for crypto.

    Available for both BTC and ETH.
    """

    underlying: str  # "BTC" or "ETH"
    value: Decimal  # Current DVOL value (annualized IV)
    timestamp_ms: int
    high_24h: Optional[Decimal] = None
    low_24h: Optional[Decimal] = None
    change_24h: Optional[Decimal] = None

    @classmethod
    def from_api_response(cls, data: Dict[str, Any], underlying: str) -> "DVOLData":
        """Create DVOL data from Deribit API response."""
        return cls(
            underlying=underlying,
            value=Decimal(str(data.get("volatility", data.get("value", 0)))),
            timestamp_ms=data.get("timestamp", int(time.time() * 1000)),
            high_24h=Decimal(str(data["high"])) if data.get("high") else None,
            low_24h=Decimal(str(data["low"])) if data.get("low") else None,
            change_24h=Decimal(str(data["change"])) if data.get("change") else None,
        )


@dataclass
class DeribitInstrumentInfo:
    """
    Detailed instrument information from Deribit.

    Includes trading parameters and current status.
    """

    instrument_name: str
    underlying: str
    expiration: date
    strike: Decimal
    option_type: DeribitOptionType
    settlement_currency: str
    contract_size: Decimal
    tick_size: Decimal
    min_trade_amount: Decimal
    maker_commission: Decimal  # e.g., 0.0003 = 0.03%
    taker_commission: Decimal  # e.g., 0.0003 = 0.03%
    is_active: bool
    creation_timestamp: int
    expiration_timestamp: int

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "DeribitInstrumentInfo":
        """Create instrument info from Deribit API response."""
        parsed = parse_deribit_instrument_name(data["instrument_name"])

        # Convert expiration timestamp to date
        exp_ts = data.get("expiration_timestamp", 0)
        if exp_ts > 0:
            expiration = datetime.fromtimestamp(exp_ts / 1000, tz=timezone.utc).date()
        else:
            expiration = parsed["expiration"]

        return cls(
            instrument_name=data["instrument_name"],
            underlying=parsed["underlying"],
            expiration=expiration,
            strike=parsed["strike"],
            option_type=parsed["option_type"],
            settlement_currency=data.get("settlement_currency", parsed["underlying"]),
            contract_size=Decimal(str(data.get("contract_size", 1))),
            tick_size=Decimal(str(data.get("tick_size", 0.0001))),
            min_trade_amount=Decimal(str(data.get("min_trade_amount", 0.1))),
            maker_commission=Decimal(str(data.get("maker_commission", 0.0003))),
            taker_commission=Decimal(str(data.get("taker_commission", 0.0003))),
            is_active=data.get("is_active", True),
            creation_timestamp=data.get("creation_timestamp", 0),
            expiration_timestamp=data.get("expiration_timestamp", 0),
        )


@dataclass
class DeribitPosition:
    """
    Position in a Deribit option.

    Size is in contracts (positive = long, negative = short).
    P&L is in the underlying cryptocurrency (inverse settlement).
    """

    instrument_name: str
    size: Decimal  # Positive = long, negative = short
    average_price: Decimal  # Entry price in underlying
    mark_price: Decimal  # Current mark price
    index_price: Decimal  # Underlying index price
    realized_pnl: Decimal  # In underlying
    unrealized_pnl: Decimal  # In underlying (floating)
    total_pnl: Decimal  # realized + unrealized
    delta: Decimal  # Position delta
    gamma: Decimal  # Position gamma
    theta: Decimal  # Position theta
    vega: Decimal  # Position vega
    maintenance_margin: Decimal  # Required margin in underlying
    initial_margin: Decimal  # Initial margin in underlying

    @property
    def is_long(self) -> bool:
        """Returns True if position is long."""
        return self.size > 0

    @property
    def is_short(self) -> bool:
        """Returns True if position is short."""
        return self.size < 0

    @property
    def notional_usd(self) -> Decimal:
        """Calculate position notional in USD."""
        return abs(self.size) * self.index_price

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "DeribitPosition":
        """Create position from Deribit API response."""
        return cls(
            instrument_name=data["instrument_name"],
            size=Decimal(str(data.get("size", 0))),
            average_price=Decimal(str(data.get("average_price", 0))),
            mark_price=Decimal(str(data.get("mark_price", 0))),
            index_price=Decimal(str(data.get("index_price", 0))),
            realized_pnl=Decimal(str(data.get("realized_profit_loss", 0))),
            unrealized_pnl=Decimal(str(data.get("floating_profit_loss", 0))),
            total_pnl=Decimal(str(data.get("total_profit_loss", 0))),
            delta=Decimal(str(data.get("delta", 0))),
            gamma=Decimal(str(data.get("gamma", 0))),
            theta=Decimal(str(data.get("theta", 0))),
            vega=Decimal(str(data.get("vega", 0))),
            maintenance_margin=Decimal(str(data.get("maintenance_margin", 0))),
            initial_margin=Decimal(str(data.get("initial_margin", 0))),
        )


@dataclass
class DeribitOrder:
    """
    Order to be submitted to Deribit.
    """

    instrument_name: str
    direction: DeribitDirection
    amount: Decimal  # In contracts
    order_type: DeribitOrderType = DeribitOrderType.LIMIT
    price: Optional[Decimal] = None  # Required for limit orders
    time_in_force: DeribitTimeInForce = DeribitTimeInForce.GTC
    reduce_only: bool = False
    post_only: bool = False  # Maker-only order
    label: Optional[str] = None  # Client order ID

    def to_api_params(self) -> Dict[str, Any]:
        """Convert to Deribit API parameters."""
        params = {
            "instrument_name": self.instrument_name,
            "amount": float(self.amount),
            "type": self.order_type.value,
        }

        if self.price is not None:
            params["price"] = float(self.price)

        if self.time_in_force != DeribitTimeInForce.GTC:
            params["time_in_force"] = self.time_in_force.value

        if self.reduce_only:
            params["reduce_only"] = True

        if self.post_only:
            params["post_only"] = True

        if self.label:
            params["label"] = self.label

        return params


@dataclass
class DeribitOrderResult:
    """
    Result of an order submission or query.
    """

    order_id: str
    instrument_name: str
    direction: DeribitDirection
    amount: Decimal
    price: Optional[Decimal]
    filled_amount: Decimal
    average_price: Optional[Decimal]
    order_state: DeribitOrderState
    order_type: DeribitOrderType
    creation_timestamp: int
    last_update_timestamp: int
    commission: Decimal  # In underlying
    label: Optional[str] = None

    @property
    def is_filled(self) -> bool:
        """Returns True if order is completely filled."""
        return self.order_state == DeribitOrderState.FILLED

    @property
    def is_open(self) -> bool:
        """Returns True if order is still active."""
        return self.order_state == DeribitOrderState.OPEN

    @property
    def remaining_amount(self) -> Decimal:
        """Calculate unfilled amount."""
        return self.amount - self.filled_amount

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "DeribitOrderResult":
        """Create order result from Deribit API response."""
        return cls(
            order_id=data["order_id"],
            instrument_name=data["instrument_name"],
            direction=DeribitDirection(data["direction"]),
            amount=Decimal(str(data["amount"])),
            price=Decimal(str(data["price"])) if data.get("price") else None,
            filled_amount=Decimal(str(data.get("filled_amount", 0))),
            average_price=(
                Decimal(str(data["average_price"])) if data.get("average_price") else None
            ),
            order_state=DeribitOrderState(data["order_state"]),
            order_type=DeribitOrderType(data["order_type"]),
            creation_timestamp=data.get("creation_timestamp", 0),
            last_update_timestamp=data.get("last_update_timestamp", 0),
            commission=Decimal(str(data.get("commission", 0))),
            label=data.get("label"),
        )


# =============================================================================
# Rate Limiter
# =============================================================================


class DeribitRateLimiter:
    """
    Rate limiter for Deribit API requests.

    Deribit rate limits:
        - 20 requests per second (REST API)
        - 100 requests per second (WebSocket)
    """

    def __init__(
        self,
        requests_per_second: float = 15.0,  # Conservative
        burst_size: int = 5,
    ):
        self._requests_per_second = requests_per_second
        self._burst_size = burst_size
        self._tokens = float(burst_size)
        self._last_update = time.monotonic()
        self._lock_time = 0.0

    def acquire(self, timeout: float = 10.0) -> bool:
        """
        Acquire a rate limit token.

        Returns True if acquired, False if timeout.
        """
        start = time.monotonic()

        while True:
            now = time.monotonic()
            elapsed = now - self._last_update
            self._tokens = min(
                float(self._burst_size), self._tokens + elapsed * self._requests_per_second
            )
            self._last_update = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True

            if now - start >= timeout:
                return False

            # Sleep for a short time
            sleep_time = (1.0 - self._tokens) / self._requests_per_second
            time.sleep(min(sleep_time, 0.1))


# =============================================================================
# API Client
# =============================================================================


class DeribitAPIClient:
    """
    Low-level client for Deribit REST API.

    Handles authentication, rate limiting, and request formatting.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        testnet: bool = True,
        timeout: float = 10.0,
    ):
        self._client_id = client_id
        self._client_secret = client_secret
        self._testnet = testnet
        self._timeout = timeout
        self._base_url = DERIBIT_TESTNET_URL if testnet else DERIBIT_API_URL
        self._rate_limiter = DeribitRateLimiter()
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0.0
        self._session = requests.Session()

    @property
    def is_authenticated(self) -> bool:
        """Check if we have valid authentication."""
        return self._access_token is not None and time.time() < self._token_expiry

    def authenticate(self) -> bool:
        """
        Authenticate with Deribit API.

        Returns True if authentication successful.
        """
        if not self._client_id or not self._client_secret:
            logger.warning("No API credentials provided, using public endpoints only")
            return False

        try:
            response = self._request(
                "public/auth",
                {
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
                authenticated=False,
            )

            if response and "access_token" in response:
                self._access_token = response["access_token"]
                # Token typically valid for 900 seconds
                self._token_expiry = time.time() + response.get("expires_in", 900) - 60
                logger.info("Deribit authentication successful")
                return True

            logger.error("Deribit authentication failed: no access token in response")
            return False

        except Exception as e:
            logger.error(f"Deribit authentication failed: {e}")
            return False

    def _request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        authenticated: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Make a request to Deribit API.

        Args:
            method: API method (e.g., "public/get_instruments")
            params: Request parameters
            authenticated: Whether this requires authentication

        Returns:
            Response data or None on error
        """
        if not self._rate_limiter.acquire():
            logger.error("Rate limit timeout")
            return None

        url = f"{self._base_url}/{method}"
        headers = {"Content-Type": "application/json"}

        if authenticated:
            if not self.is_authenticated:
                if not self.authenticate():
                    raise RuntimeError("Authentication required but failed")
            headers["Authorization"] = f"Bearer {self._access_token}"

        try:
            payload = {"jsonrpc": "2.0", "id": 1, "method": method}
            if params:
                payload["params"] = params

            response = self._session.post(
                url,
                json=payload,
                headers=headers,
                timeout=self._timeout,
            )
            response.raise_for_status()

            data = response.json()

            if "error" in data:
                error = data["error"]
                logger.error(f"Deribit API error: {error}")
                return None

            return data.get("result")

        except requests.exceptions.RequestException as e:
            logger.error(f"Deribit request failed: {e}")
            return None

    def get_instruments(
        self,
        currency: str,
        kind: str = "option",
        expired: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get available instruments."""
        result = self._request(
            "public/get_instruments",
            {
                "currency": currency,
                "kind": kind,
                "expired": expired,
            },
        )
        return result or []

    def get_ticker(self, instrument_name: str) -> Optional[Dict[str, Any]]:
        """Get ticker/quote for an instrument."""
        return self._request(
            "public/ticker",
            {"instrument_name": instrument_name},
        )

    def get_order_book(
        self,
        instrument_name: str,
        depth: int = 10,
    ) -> Optional[Dict[str, Any]]:
        """Get order book for an instrument."""
        return self._request(
            "public/get_order_book",
            {
                "instrument_name": instrument_name,
                "depth": depth,
            },
        )

    def get_index_price(self, currency: str) -> Optional[Dict[str, Any]]:
        """Get current index price for currency."""
        return self._request(
            "public/get_index_price",
            {"index_name": f"{currency.lower()}_usd"},
        )

    def get_volatility_index(self, currency: str) -> Optional[Dict[str, Any]]:
        """Get DVOL (volatility index) for currency."""
        return self._request(
            "public/get_volatility_index_data",
            {"currency": currency, "resolution": "1"},
        )

    def buy(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Submit buy order."""
        return self._request("private/buy", params, authenticated=True)

    def sell(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Submit sell order."""
        return self._request("private/sell", params, authenticated=True)

    def cancel(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Cancel an order."""
        return self._request(
            "private/cancel",
            {"order_id": order_id},
            authenticated=True,
        )

    def cancel_all_by_instrument(self, instrument_name: str) -> Optional[int]:
        """Cancel all orders for an instrument. Returns count."""
        result = self._request(
            "private/cancel_all_by_instrument",
            {"instrument_name": instrument_name},
            authenticated=True,
        )
        return result if isinstance(result, int) else None

    def get_positions(self, currency: str) -> List[Dict[str, Any]]:
        """Get all positions for a currency."""
        result = self._request(
            "private/get_positions",
            {"currency": currency},
            authenticated=True,
        )
        return result or []

    def get_open_orders(self, currency: str) -> List[Dict[str, Any]]:
        """Get all open orders for a currency."""
        result = self._request(
            "private/get_open_orders_by_currency",
            {"currency": currency},
            authenticated=True,
        )
        return result or []

    def get_order_state(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get current state of an order."""
        return self._request(
            "private/get_order_state",
            {"order_id": order_id},
            authenticated=True,
        )

    def get_account_summary(self, currency: str) -> Optional[Dict[str, Any]]:
        """Get account summary for a currency."""
        return self._request(
            "private/get_account_summary",
            {"currency": currency, "extended": True},
            authenticated=True,
        )


# =============================================================================
# Market Data Adapter
# =============================================================================


class DeribitOptionsMarketDataAdapter(MarketDataAdapter):
    """
    Market data adapter for Deribit options.

    Provides:
        - Option chain retrieval
        - Real-time quotes with Greeks
        - Order book data
        - DVOL (volatility index)
        - Index prices
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.DERIBIT,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._vendor = vendor
        self._config = config or {}
        self._client = DeribitAPIClient(
            client_id=self._config.get("client_id"),
            client_secret=self._config.get("client_secret"),
            testnet=self._config.get("testnet", True),
            timeout=self._config.get("timeout", 10.0),
        )
        self._contract_cache: Dict[str, DeribitOptionContract] = {}
        self._quote_cache: Dict[str, DeribitOptionQuote] = {}
        self._cache_ttl = self._config.get("cache_ttl_sec", 5.0)
        self._cache_times: Dict[str, float] = {}

    @property
    def vendor(self) -> ExchangeVendor:
        return self._vendor

    @property
    def market_type(self) -> MarketType:
        return MarketType.CRYPTO_OPTIONS

    def get_option_chain(
        self,
        underlying: str,
        expiration: Optional[date] = None,
    ) -> List[DeribitOptionContract]:
        """
        Get option chain for an underlying.

        Args:
            underlying: "BTC" or "ETH"
            expiration: Optional specific expiration date

        Returns:
            List of option contracts
        """
        underlying = underlying.upper()
        instruments = self._client.get_instruments(underlying, kind="option")

        contracts = []
        for inst in instruments:
            try:
                contract = DeribitOptionContract.from_api_response(inst)

                # Filter by expiration if specified
                if expiration is not None and contract.expiration != expiration:
                    continue

                # Cache contract
                self._contract_cache[contract.instrument_name] = contract
                contracts.append(contract)

            except Exception as e:
                logger.warning(f"Failed to parse instrument {inst.get('instrument_name')}: {e}")

        return contracts

    def get_expirations(self, underlying: str) -> List[date]:
        """
        Get available expiration dates for an underlying.

        Returns sorted list of expiration dates.
        """
        contracts = self.get_option_chain(underlying)
        expirations = sorted(set(c.expiration for c in contracts))
        return expirations

    def get_strikes(
        self,
        underlying: str,
        expiration: date,
    ) -> List[Decimal]:
        """
        Get available strikes for a specific expiration.

        Returns sorted list of strike prices.
        """
        contracts = self.get_option_chain(underlying, expiration)
        strikes = sorted(set(c.strike for c in contracts))
        return strikes

    def get_option_quote(
        self,
        instrument_name: str,
        use_cache: bool = True,
    ) -> Optional[DeribitOptionQuote]:
        """
        Get real-time quote for an option.

        Args:
            instrument_name: Deribit instrument name
            use_cache: Whether to use cached data if fresh

        Returns:
            Option quote or None on error
        """
        # Check cache
        if use_cache:
            cache_time = self._cache_times.get(f"quote:{instrument_name}", 0)
            if time.time() - cache_time < self._cache_ttl:
                cached = self._quote_cache.get(instrument_name)
                if cached is not None:
                    return cached

        # Fetch fresh data
        data = self._client.get_ticker(instrument_name)
        if data is None:
            return None

        try:
            quote = DeribitOptionQuote.from_api_response(data)
            self._quote_cache[instrument_name] = quote
            self._cache_times[f"quote:{instrument_name}"] = time.time()
            return quote
        except Exception as e:
            logger.error(f"Failed to parse quote for {instrument_name}: {e}")
            return None

    def get_option_quotes_batch(
        self,
        instrument_names: List[str],
    ) -> Dict[str, DeribitOptionQuote]:
        """
        Get quotes for multiple options.

        Note: Deribit doesn't have batch ticker endpoint,
        so this makes sequential requests with rate limiting.
        """
        quotes = {}
        for name in instrument_names:
            quote = self.get_option_quote(name, use_cache=False)
            if quote is not None:
                quotes[name] = quote
        return quotes

    def get_orderbook(
        self,
        instrument_name: str,
        depth: int = 10,
    ) -> Optional[DeribitOrderbook]:
        """Get order book for an option."""
        data = self._client.get_order_book(instrument_name, depth)
        if data is None:
            return None

        try:
            return DeribitOrderbook.from_api_response(data)
        except Exception as e:
            logger.error(f"Failed to parse orderbook for {instrument_name}: {e}")
            return None

    def get_dvol(self, underlying: str) -> Optional[DVOLData]:
        """
        Get DVOL (30-day volatility index) for an underlying.

        Args:
            underlying: "BTC" or "ETH"

        Returns:
            DVOL data or None on error
        """
        data = self._client.get_volatility_index(underlying.upper())
        if data is None or not data.get("data"):
            return None

        try:
            # Get most recent data point
            latest = data["data"][-1] if isinstance(data["data"], list) else data["data"]
            return DVOLData.from_api_response(latest, underlying.upper())
        except Exception as e:
            logger.error(f"Failed to parse DVOL for {underlying}: {e}")
            return None

    def get_index_price(self, underlying: str) -> Optional[Decimal]:
        """Get current index (spot) price for underlying."""
        data = self._client.get_index_price(underlying.upper())
        if data is None:
            return None

        try:
            return Decimal(str(data.get("index_price", 0)))
        except Exception as e:
            logger.error(f"Failed to parse index price for {underlying}: {e}")
            return None

    def get_atm_strike(
        self,
        underlying: str,
        expiration: date,
    ) -> Optional[Decimal]:
        """
        Get at-the-money strike for given expiration.

        Returns the strike closest to current index price.
        """
        index_price = self.get_index_price(underlying)
        if index_price is None:
            return None

        strikes = self.get_strikes(underlying, expiration)
        if not strikes:
            return None

        # Find closest strike
        return min(strikes, key=lambda s: abs(s - index_price))

    # Required abstract method implementations from MarketDataAdapter
    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        *,
        limit: int = 500,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> List[Bar]:
        """
        Not applicable for options - use get_option_quote instead.

        Options don't have traditional OHLCV bars.
        Use get_option_quote() for price data.
        """
        raise NotImplementedError(
            "Options do not have OHLCV bars. Use get_option_quote() for price data."
        )

    def get_latest_bar(self, symbol: str, timeframe: str) -> Optional[Bar]:
        """
        Not applicable for options.

        Options don't have traditional OHLCV bars.
        Use get_option_quote() for price data.
        """
        raise NotImplementedError(
            "Options do not have OHLCV bars. Use get_option_quote() for price data."
        )

    def get_tick(self, symbol: str) -> Optional[Tick]:
        """
        Get latest tick for an option.

        Maps to get_option_quote and returns a Tick with bid/ask/last.
        """
        quote = self.get_option_quote(symbol)
        if quote is None:
            return None

        # Convert to Tick format
        return Tick(
            ts=quote.timestamp_ms,
            symbol=quote.instrument_name,
            bid=float(quote.bid_price) if quote.bid_price else None,
            ask=float(quote.ask_price) if quote.ask_price else None,
            last=float(quote.last_price) if quote.last_price else None,
            bid_size=float(quote.bid_size) if quote.bid_size else None,
            ask_size=float(quote.ask_size) if quote.ask_size else None,
        )

    def stream_bars(
        self,
        symbols: List[str],
        interval_ms: int,
    ) -> Iterator[Bar]:
        """
        Not applicable for options.

        Options don't have traditional OHLCV bars.
        Use WebSocket streaming with DeribitWebSocketClient for real-time quotes.
        """
        raise NotImplementedError(
            "Options do not support bar streaming. "
            "Use DeribitWebSocketClient for real-time quote streaming."
        )

    def stream_ticks(
        self,
        symbols: List[str],
    ) -> Iterator[Tick]:
        """
        Not implemented in REST adapter.

        For real-time tick streaming, use DeribitWebSocketClient
        with subscribe_ticker() method.
        """
        raise NotImplementedError(
            "Real-time tick streaming requires WebSocket connection. "
            "Use DeribitWebSocketClient for streaming data."
        )


# =============================================================================
# Order Execution Adapter
# =============================================================================


class DeribitOptionsOrderExecutionAdapter(OrderExecutionAdapter):
    """
    Order execution adapter for Deribit options.

    Provides:
        - Order submission (market, limit)
        - Order cancellation
        - Position management
        - Account summary
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.DERIBIT,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._vendor = vendor
        self._config = config or {}
        self._client = DeribitAPIClient(
            client_id=self._config.get("client_id"),
            client_secret=self._config.get("client_secret"),
            testnet=self._config.get("testnet", True),
            timeout=self._config.get("timeout", 10.0),
        )
        # Authenticate on creation
        if self._config.get("client_id"):
            self._client.authenticate()

    @property
    def vendor(self) -> ExchangeVendor:
        return self._vendor

    @property
    def market_type(self) -> MarketType:
        return MarketType.CRYPTO_OPTIONS

    def submit_order(self, order: DeribitOrder) -> Optional[DeribitOrderResult]:
        """
        Submit an order to Deribit.

        Args:
            order: Order to submit

        Returns:
            Order result or None on error
        """
        params = order.to_api_params()

        if order.direction == DeribitDirection.BUY:
            response = self._client.buy(params)
        else:
            response = self._client.sell(params)

        if response is None:
            return None

        try:
            # Response contains 'order' key with order details
            order_data = response.get("order", response)
            return DeribitOrderResult.from_api_response(order_data)
        except Exception as e:
            logger.error(f"Failed to parse order response: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Returns True if cancellation successful.
        """
        result = self._client.cancel(order_id)
        return result is not None

    def cancel_all(self, instrument_name: str) -> int:
        """
        Cancel all orders for an instrument.

        Returns number of orders cancelled.
        """
        result = self._client.cancel_all_by_instrument(instrument_name)
        return result or 0

    def get_order(self, order_id: str) -> Optional[DeribitOrderResult]:
        """Get current state of an order."""
        data = self._client.get_order_state(order_id)
        if data is None:
            return None

        try:
            return DeribitOrderResult.from_api_response(data)
        except Exception as e:
            logger.error(f"Failed to parse order state: {e}")
            return None

    def get_open_orders(self, currency: str) -> List[DeribitOrderResult]:
        """Get all open orders for a currency."""
        orders_data = self._client.get_open_orders(currency.upper())

        orders = []
        for data in orders_data:
            try:
                orders.append(DeribitOrderResult.from_api_response(data))
            except Exception as e:
                logger.warning(f"Failed to parse order: {e}")

        return orders

    def get_positions(self, currency: str) -> List[DeribitPosition]:
        """Get all positions for a currency."""
        positions_data = self._client.get_positions(currency.upper())

        positions = []
        for data in positions_data:
            try:
                # Only include positions with non-zero size
                if Decimal(str(data.get("size", 0))) != 0:
                    positions.append(DeribitPosition.from_api_response(data))
            except Exception as e:
                logger.warning(f"Failed to parse position: {e}")

        return positions

    def get_position(self, instrument_name: str) -> Optional[DeribitPosition]:
        """Get position for a specific instrument."""
        parsed = parse_deribit_instrument_name(instrument_name)
        currency = parsed["underlying"]

        positions = self.get_positions(currency)
        for pos in positions:
            if pos.instrument_name == instrument_name:
                return pos

        return None

    def get_account_summary(self, currency: str) -> Optional[Dict[str, Any]]:
        """
        Get account summary.

        Returns dict with:
            - equity: Total account equity
            - balance: Cash balance
            - margin_balance: Available margin
            - initial_margin: Used initial margin
            - maintenance_margin: Used maintenance margin
            - available_funds: Available for trading
            - delta_total: Portfolio delta
            - options_value: Mark-to-market options value
        """
        data = self._client.get_account_summary(currency.upper())
        if data is None:
            return None

        return {
            "equity": Decimal(str(data.get("equity", 0))),
            "balance": Decimal(str(data.get("balance", 0))),
            "margin_balance": Decimal(str(data.get("margin_balance", 0))),
            "initial_margin": Decimal(str(data.get("initial_margin", 0))),
            "maintenance_margin": Decimal(str(data.get("maintenance_margin", 0))),
            "available_funds": Decimal(str(data.get("available_funds", 0))),
            "delta_total": Decimal(str(data.get("delta_total", 0))),
            "options_value": Decimal(str(data.get("options_value", 0))),
            "currency": currency.upper(),
        }

    # =========================================================================
    # Abstract Method Implementations (required by OrderExecutionAdapter)
    # =========================================================================

    def get_order_status(
        self,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> Optional[ExecReport]:
        """
        Get current status of an order.

        Args:
            order_id: Exchange order ID
            client_order_id: Client order ID (not supported by Deribit, use order_id)

        Returns:
            ExecReport with current status, or None if not found

        Note:
            Deribit uses exchange order IDs only. client_order_id is ignored.
            Use get_order() for Deribit-specific DeribitOrderResult.
        """
        if order_id is None:
            logger.warning("get_order_status requires order_id for Deribit")
            return None

        deribit_order = self.get_order(order_id)
        if deribit_order is None:
            return None

        # Map Deribit order state to ExecStatus
        status_mapping = {
            "open": ExecStatus.NEW,
            "filled": ExecStatus.FILLED,
            "cancelled": ExecStatus.CANCELED,
            "rejected": ExecStatus.REJECTED,
            "untriggered": ExecStatus.NEW,
        }
        exec_status = status_mapping.get(
            deribit_order.state.lower() if deribit_order.state else "open",
            ExecStatus.NEW,
        )

        # Handle partially filled
        if deribit_order.filled_amount and deribit_order.filled_amount > 0:
            if deribit_order.amount and deribit_order.filled_amount < deribit_order.amount:
                exec_status = ExecStatus.PARTIALLY_FILLED

        # Map direction to Side
        side = CoreSide.BUY if deribit_order.direction == DeribitDirection.BUY else CoreSide.SELL

        # Map order type
        order_type = CoreOrderType.LIMIT
        if deribit_order.order_type == DeribitOrderType.MARKET:
            order_type = CoreOrderType.MARKET

        return ExecReport(
            ts=int(deribit_order.creation_timestamp or 0),
            run_id="deribit",
            symbol=deribit_order.instrument_name or "",
            side=side,
            order_type=order_type,
            price=deribit_order.average_price or deribit_order.price or Decimal("0"),
            quantity=deribit_order.filled_amount or Decimal("0"),
            fee=deribit_order.commission or Decimal("0"),
            fee_asset=deribit_order.fee_currency,
            exec_status=exec_status,
            liquidity=Liquidity.UNKNOWN,
            order_id=deribit_order.order_id,
            client_order_id=None,
        )

    def get_account_info(self) -> AccountInfo:
        """
        Get account information.

        Returns:
            AccountInfo with balances and status

        Note:
            For Deribit, this returns BTC account summary by default.
            Use get_account_summary(currency) for specific currencies.
        """
        # Default to BTC account for options trading
        summary = self.get_account_summary("BTC")

        if summary is None:
            # Return empty AccountInfo if API call fails
            return AccountInfo(
                vendor=self._vendor,
                account_id="",
                account_type="options",
                cash_balance=Decimal("0"),
                buying_power=Decimal("0"),
            )

        return AccountInfo(
            vendor=self._vendor,
            account_id=str(summary.get("currency", "BTC")),
            account_type="options",
            cash_balance=summary.get("balance", Decimal("0")),
            buying_power=summary.get("available_funds", Decimal("0")),
            margin_enabled=True,
            raw_data=summary,
        )


# =============================================================================
# Factory Functions
# =============================================================================


def create_deribit_options_market_data_adapter(
    testnet: bool = True,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    **kwargs,
) -> DeribitOptionsMarketDataAdapter:
    """
    Create a Deribit options market data adapter.

    Args:
        testnet: If True, use testnet (paper trading)
        client_id: Optional API client ID
        client_secret: Optional API client secret
        **kwargs: Additional configuration

    Returns:
        Configured market data adapter

    Example:
        >>> adapter = create_deribit_options_market_data_adapter(testnet=True)
        >>> chain = adapter.get_option_chain("BTC")
        >>> dvol = adapter.get_dvol("BTC")
    """
    config = {
        "testnet": testnet,
        "client_id": client_id,
        "client_secret": client_secret,
        **kwargs,
    }
    return DeribitOptionsMarketDataAdapter(
        vendor=ExchangeVendor.DERIBIT,
        config=config,
    )


def create_deribit_options_order_execution_adapter(
    client_id: str,
    client_secret: str,
    testnet: bool = True,
    **kwargs,
) -> DeribitOptionsOrderExecutionAdapter:
    """
    Create a Deribit options order execution adapter.

    Args:
        client_id: API client ID (required for trading)
        client_secret: API client secret (required for trading)
        testnet: If True, use testnet (paper trading)
        **kwargs: Additional configuration

    Returns:
        Configured order execution adapter

    Example:
        >>> adapter = create_deribit_options_order_execution_adapter(
        ...     client_id="...",
        ...     client_secret="...",
        ...     testnet=True,
        ... )
        >>> order = DeribitOrder(
        ...     instrument_name="BTC-28MAR25-100000-C",
        ...     direction=DeribitDirection.BUY,
        ...     amount=Decimal("0.1"),
        ...     order_type=DeribitOrderType.LIMIT,
        ...     price=Decimal("0.05"),
        ... )
        >>> result = adapter.submit_order(order)
    """
    config = {
        "testnet": testnet,
        "client_id": client_id,
        "client_secret": client_secret,
        **kwargs,
    }
    return DeribitOptionsOrderExecutionAdapter(
        vendor=ExchangeVendor.DERIBIT,
        config=config,
    )
