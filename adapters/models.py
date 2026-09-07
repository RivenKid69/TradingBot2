# -*- coding: utf-8 -*-
"""
adapters/models.py
Exchange-agnostic data models for multi-exchange support.

These models provide a unified interface for different asset classes (crypto, stocks)
and exchanges (Binance, Alpaca, etc.) while maintaining compatibility with existing
core_models.py entities.

Design Principles:
- Immutable dataclasses for thread safety and clarity
- Decimal for financial precision (prices, quantities)
- int milliseconds UTC for timestamps
- Backward compatible with existing core_models.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Mapping, Tuple
import json


# =========================
# Enumerations
# =========================


class MarketType(str, Enum):
    """Type of market/asset class."""

    # Crypto
    CRYPTO_SPOT = "CRYPTO_SPOT"
    CRYPTO_FUTURES = "CRYPTO_FUTURES"
    CRYPTO_PERP = "CRYPTO_PERP"
    CRYPTO_OPTIONS = "CRYPTO_OPTIONS"  # Deribit BTC/ETH options
    # Equity
    EQUITY = "EQUITY"
    EQUITY_OPTIONS = "EQUITY_OPTIONS"
    # Forex
    FOREX = "FOREX"
    # Traditional Futures (CME, CBOT, NYMEX, COMEX)
    INDEX_FUTURES = "INDEX_FUTURES"  # ES, NQ, YM, RTY
    COMMODITY_FUTURES = "COMMODITY_FUTURES"  # GC, SI, CL, NG, ZC, ZS, ZW
    CURRENCY_FUTURES = "CURRENCY_FUTURES"  # 6E, 6J, 6B, 6A, 6C, 6S
    BOND_FUTURES = "BOND_FUTURES"  # ZN, ZB, ZT, ZF, UB

    @property
    def is_crypto(self) -> bool:
        return self in (
            MarketType.CRYPTO_SPOT,
            MarketType.CRYPTO_FUTURES,
            MarketType.CRYPTO_PERP,
            MarketType.CRYPTO_OPTIONS,
        )

    @property
    def is_equity(self) -> bool:
        return self in (MarketType.EQUITY, MarketType.EQUITY_OPTIONS)

    @property
    def is_futures(self) -> bool:
        """Returns True if this is any type of futures contract."""
        return self in (
            MarketType.CRYPTO_FUTURES,
            MarketType.CRYPTO_PERP,
            MarketType.INDEX_FUTURES,
            MarketType.COMMODITY_FUTURES,
            MarketType.CURRENCY_FUTURES,
            MarketType.BOND_FUTURES,
        )

    @property
    def is_traditional_futures(self) -> bool:
        """Returns True if this is a traditional (non-crypto) futures contract."""
        return self in (
            MarketType.INDEX_FUTURES,
            MarketType.COMMODITY_FUTURES,
            MarketType.CURRENCY_FUTURES,
            MarketType.BOND_FUTURES,
        )

    @property
    def has_expiry(self) -> bool:
        """Returns True if contracts have expiry dates (not perpetuals)."""
        return self in (
            MarketType.CRYPTO_FUTURES,
            MarketType.CRYPTO_OPTIONS,
            MarketType.INDEX_FUTURES,
            MarketType.COMMODITY_FUTURES,
            MarketType.CURRENCY_FUTURES,
            MarketType.BOND_FUTURES,
            MarketType.EQUITY_OPTIONS,
        )

    @property
    def has_funding_rate(self) -> bool:
        """Returns True if this market type has funding rate mechanism."""
        return self == MarketType.CRYPTO_PERP

    @property
    def has_trading_hours(self) -> bool:
        """Returns True if this market type has trading hours (not 24/7)."""
        return self.is_equity or self == MarketType.FOREX or self.is_traditional_futures


class ExchangeVendor(str, Enum):
    """Supported exchange vendors."""

    # Crypto Spot
    BINANCE = "binance"
    BINANCE_US = "binance_us"
    # Crypto Futures
    BINANCE_FUTURES = "binance_futures"  # Binance USDT-M Futures
    BINANCE_COIN_FUTURES = "binance_coin_futures"  # Binance COIN-M Futures
    # Equity
    ALPACA = "alpaca"
    POLYGON = "polygon"  # Data provider
    YAHOO = "yahoo"  # Data provider for corporate actions, earnings (Phase 7)
    # Options Data Providers (Phase 2)
    THETA_DATA = "theta_data"  # Theta Data - US options (historical + delayed real-time)
    # Crypto Options (Phase 2B)
    DERIBIT = "deribit"  # Deribit - BTC/ETH options (European, inverse settlement)
    # Forex (Phase 0)
    OANDA = "oanda"  # Primary forex broker (OANDA v20 API)
    IG = "ig"  # IG Markets (alternative)
    DUKASCOPY = "dukascopy"  # Dukascopy (historical tick data)
    # Traditional Futures (via Interactive Brokers)
    IB = "ib"  # Interactive Brokers (CME, CBOT, NYMEX, COMEX)
    IB_CME = "ib_cme"  # IB → CME (ES, NQ, 6E, 6J)
    IB_CBOT = "ib_cbot"  # IB → CBOT (ZN, ZB, ZC, ZS, ZW)
    IB_NYMEX = "ib_nymex"  # IB → NYMEX (CL, NG)
    IB_COMEX = "ib_comex"  # IB → COMEX (GC, SI)
    # Unknown
    UNKNOWN = "unknown"

    @property
    def market_type(self) -> MarketType:
        """Default market type for vendor."""
        if self in (ExchangeVendor.BINANCE, ExchangeVendor.BINANCE_US):
            return MarketType.CRYPTO_SPOT
        elif self in (ExchangeVendor.BINANCE_FUTURES, ExchangeVendor.BINANCE_COIN_FUTURES):
            return MarketType.CRYPTO_PERP
        elif self == ExchangeVendor.DERIBIT:
            return MarketType.CRYPTO_OPTIONS
        elif self == ExchangeVendor.ALPACA:
            return MarketType.EQUITY
        elif self == ExchangeVendor.THETA_DATA:
            return MarketType.EQUITY_OPTIONS
        elif self in (ExchangeVendor.OANDA, ExchangeVendor.IG, ExchangeVendor.DUKASCOPY):
            return MarketType.FOREX
        elif self in (ExchangeVendor.IB, ExchangeVendor.IB_CME):
            return MarketType.INDEX_FUTURES
        elif self == ExchangeVendor.IB_CBOT:
            return MarketType.BOND_FUTURES
        elif self == ExchangeVendor.IB_NYMEX:
            return MarketType.COMMODITY_FUTURES
        elif self == ExchangeVendor.IB_COMEX:
            return MarketType.COMMODITY_FUTURES
        return MarketType.CRYPTO_SPOT

    @property
    def is_forex(self) -> bool:
        """Returns True if this vendor is a forex broker/data provider."""
        return self in (ExchangeVendor.OANDA, ExchangeVendor.IG, ExchangeVendor.DUKASCOPY)

    @property
    def is_crypto_futures(self) -> bool:
        """Returns True if this vendor supports crypto futures."""
        return self in (ExchangeVendor.BINANCE_FUTURES, ExchangeVendor.BINANCE_COIN_FUTURES)

    @property
    def is_crypto_options(self) -> bool:
        """Returns True if this vendor supports crypto options."""
        return self == ExchangeVendor.DERIBIT

    @property
    def is_traditional_futures(self) -> bool:
        """Returns True if this vendor supports traditional futures."""
        return self in (
            ExchangeVendor.IB,
            ExchangeVendor.IB_CME,
            ExchangeVendor.IB_CBOT,
            ExchangeVendor.IB_NYMEX,
            ExchangeVendor.IB_COMEX,
        )


class FeeStructure(str, Enum):
    """Fee calculation structure type."""

    PERCENTAGE = "percentage"  # Fee as % of notional (e.g., 0.1% = 10 bps)
    PER_SHARE = "per_share"  # Fee per share/unit (e.g., $0.005/share)
    FLAT = "flat"  # Flat fee per trade (e.g., $1.00)
    TIERED = "tiered"  # Tiered based on volume
    MIXED = "mixed"  # Combination of above


class SessionType(str, Enum):
    """Trading session type."""

    REGULAR = "regular"  # Regular market hours (9:30-16:00 ET for US)
    PRE_MARKET = "pre_market"  # Pre-market (4:00-9:30 ET)
    AFTER_HOURS = "after_hours"  # After hours (16:00-20:00 ET)
    EXTENDED = "extended"  # All extended hours
    CONTINUOUS = "continuous"  # 24/7 (crypto)


# =========================
# Exchange Rules
# =========================


@dataclass(frozen=True)
class ExchangeRule:
    """
    Trading rules and constraints for a symbol on an exchange.

    Generalizes Binance's filter concept to support multiple exchanges.
    Designed to be exchange-agnostic while capturing key trading constraints.

    Attributes:
        symbol: Trading symbol (e.g., "BTCUSDT", "AAPL")
        tick_size: Minimum price increment (e.g., 0.01)
        step_size: Minimum quantity increment (e.g., 0.001)
        min_notional: Minimum order value in quote currency
        min_qty: Minimum order quantity
        max_qty: Maximum order quantity
        max_notional: Maximum order value (optional)
        price_precision: Decimal places for price
        qty_precision: Decimal places for quantity
        market_type: Type of market (crypto, equity)
        base_asset: Base asset symbol (e.g., "BTC", "AAPL")
        quote_asset: Quote asset symbol (e.g., "USDT", "USD")
        lot_size: Standard lot size (1 for most, 100 for some options)
        is_tradable: Whether symbol is currently tradable
        is_marginable: Whether margin trading is allowed
        is_shortable: Whether short selling is allowed
        raw_filters: Original exchange-specific filter data
    """

    symbol: str
    tick_size: Decimal
    step_size: Decimal
    min_notional: Decimal
    min_qty: Decimal = Decimal("0")
    max_qty: Optional[Decimal] = None
    max_notional: Optional[Decimal] = None
    price_precision: int = 8
    qty_precision: int = 8
    market_type: MarketType = MarketType.CRYPTO_SPOT
    base_asset: str = ""
    quote_asset: str = ""
    lot_size: int = 1
    is_tradable: bool = True
    is_marginable: bool = False
    is_shortable: bool = False
    raw_filters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate immutable constraints (done via object.__setattr__ for frozen)
        if self.tick_size <= Decimal("0"):
            object.__setattr__(self, "tick_size", Decimal("0.00000001"))
        if self.step_size <= Decimal("0"):
            object.__setattr__(self, "step_size", Decimal("0.00000001"))

    def quantize_price(self, price: Decimal) -> Decimal:
        """Round price to valid tick size."""
        if self.tick_size <= 0:
            return price
        return (price // self.tick_size) * self.tick_size

    def quantize_qty(self, qty: Decimal) -> Decimal:
        """Round quantity to valid step size."""
        if self.step_size <= 0:
            return qty
        return (qty // self.step_size) * self.step_size

    def validate_order(self, price: Decimal, qty: Decimal) -> Tuple[bool, Optional[str]]:
        """
        Validate order against exchange rules.

        Returns:
            (is_valid, error_message)
        """
        notional = price * qty

        if qty < self.min_qty:
            return False, f"Quantity {qty} below minimum {self.min_qty}"

        if self.max_qty is not None and qty > self.max_qty:
            return False, f"Quantity {qty} exceeds maximum {self.max_qty}"

        if notional < self.min_notional:
            return False, f"Notional {notional} below minimum {self.min_notional}"

        if self.max_notional is not None and notional > self.max_notional:
            return False, f"Notional {notional} exceeds maximum {self.max_notional}"

        if not self.is_tradable:
            return False, f"Symbol {self.symbol} is not tradable"

        return True, None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "tick_size": str(self.tick_size),
            "step_size": str(self.step_size),
            "min_notional": str(self.min_notional),
            "min_qty": str(self.min_qty),
            "max_qty": str(self.max_qty) if self.max_qty else None,
            "max_notional": str(self.max_notional) if self.max_notional else None,
            "price_precision": self.price_precision,
            "qty_precision": self.qty_precision,
            "market_type": self.market_type.value,
            "base_asset": self.base_asset,
            "quote_asset": self.quote_asset,
            "lot_size": self.lot_size,
            "is_tradable": self.is_tradable,
            "is_marginable": self.is_marginable,
            "is_shortable": self.is_shortable,
            "raw_filters": self.raw_filters,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ExchangeRule":
        def _to_decimal(v: Any, default: str = "0") -> Decimal:
            if v is None:
                return Decimal(default)
            try:
                return Decimal(str(v))
            except Exception:
                return Decimal(default)

        def _to_decimal_opt(v: Any) -> Optional[Decimal]:
            if v is None:
                return None
            try:
                return Decimal(str(v))
            except Exception:
                return None

        market_type_val = d.get("market_type", "CRYPTO_SPOT")
        if isinstance(market_type_val, MarketType):
            market_type = market_type_val
        else:
            try:
                market_type = MarketType(str(market_type_val))
            except ValueError:
                market_type = MarketType.CRYPTO_SPOT

        return cls(
            symbol=str(d.get("symbol", "")),
            tick_size=_to_decimal(d.get("tick_size"), "0.00000001"),
            step_size=_to_decimal(d.get("step_size"), "0.00000001"),
            min_notional=_to_decimal(d.get("min_notional"), "0"),
            min_qty=_to_decimal(d.get("min_qty"), "0"),
            max_qty=_to_decimal_opt(d.get("max_qty")),
            max_notional=_to_decimal_opt(d.get("max_notional")),
            price_precision=int(d.get("price_precision", 8)),
            qty_precision=int(d.get("qty_precision", 8)),
            market_type=market_type,
            base_asset=str(d.get("base_asset", "")),
            quote_asset=str(d.get("quote_asset", "")),
            lot_size=int(d.get("lot_size", 1)),
            is_tradable=bool(d.get("is_tradable", True)),
            is_marginable=bool(d.get("is_marginable", False)),
            is_shortable=bool(d.get("is_shortable", False)),
            raw_filters=dict(d.get("raw_filters", {})),
        )


# =========================
# Trading Sessions
# =========================


@dataclass(frozen=True)
class TradingSession:
    """
    Represents a trading session with start/end times.

    Times are stored as minutes from midnight UTC.
    For timezone-aware sessions (like US market), conversion should happen
    at the adapter level.

    Attributes:
        session_type: Type of session (regular, pre-market, etc.)
        start_minutes: Start time as minutes from midnight (0-1439)
        end_minutes: End time as minutes from midnight (0-1439)
        timezone: Timezone string (e.g., "America/New_York")
        days_of_week: Days when session is active (0=Monday, 6=Sunday)
        is_active: Whether session is currently active
    """

    session_type: SessionType
    start_minutes: int  # Minutes from midnight (0-1439)
    end_minutes: int  # Minutes from midnight (0-1439)
    timezone: str = "UTC"
    days_of_week: Tuple[int, ...] = (0, 1, 2, 3, 4)  # Mon-Fri by default
    is_active: bool = True

    @property
    def start_time_str(self) -> str:
        """Returns start time as HH:MM string."""
        h, m = divmod(self.start_minutes, 60)
        return f"{h:02d}:{m:02d}"

    @property
    def end_time_str(self) -> str:
        """Returns end time as HH:MM string."""
        h, m = divmod(self.end_minutes, 60)
        return f"{h:02d}:{m:02d}"

    @property
    def duration_minutes(self) -> int:
        """Duration of session in minutes."""
        if self.end_minutes > self.start_minutes:
            return self.end_minutes - self.start_minutes
        # Overnight session
        return (1440 - self.start_minutes) + self.end_minutes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_type": self.session_type.value,
            "start_minutes": self.start_minutes,
            "end_minutes": self.end_minutes,
            "timezone": self.timezone,
            "days_of_week": list(self.days_of_week),
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TradingSession":
        session_type_val = d.get("session_type", "regular")
        try:
            session_type = SessionType(str(session_type_val))
        except ValueError:
            session_type = SessionType.REGULAR

        days = d.get("days_of_week", [0, 1, 2, 3, 4])
        if isinstance(days, (list, tuple)):
            days_tuple = tuple(int(x) for x in days)
        else:
            days_tuple = (0, 1, 2, 3, 4)

        return cls(
            session_type=session_type,
            start_minutes=int(d.get("start_minutes", 0)),
            end_minutes=int(d.get("end_minutes", 0)),
            timezone=str(d.get("timezone", "UTC")),
            days_of_week=days_tuple,
            is_active=bool(d.get("is_active", True)),
        )


@dataclass(frozen=True)
class MarketCalendar:
    """
    Market calendar with trading sessions and holidays.

    Attributes:
        vendor: Exchange vendor
        market_type: Type of market
        sessions: List of trading sessions
        holidays: List of holiday dates as (year, month, day) tuples
        half_days: List of half-day dates with early close
        timezone: Primary timezone for this market
    """

    vendor: ExchangeVendor
    market_type: MarketType
    sessions: List[TradingSession] = field(default_factory=list)
    holidays: List[Tuple[int, int, int]] = field(default_factory=list)  # (year, month, day)
    half_days: List[Tuple[int, int, int]] = field(default_factory=list)
    timezone: str = "UTC"

    @property
    def is_24_7(self) -> bool:
        """Returns True if market trades 24/7."""
        return self.market_type.is_crypto

    def get_regular_session(self) -> Optional[TradingSession]:
        """Returns the regular trading session if available."""
        for session in self.sessions:
            if session.session_type == SessionType.REGULAR:
                return session
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vendor": self.vendor.value,
            "market_type": self.market_type.value,
            "sessions": [s.to_dict() for s in self.sessions],
            "holidays": list(self.holidays),
            "half_days": list(self.half_days),
            "timezone": self.timezone,
        }


# =========================
# Fee Structures
# =========================


@dataclass(frozen=True)
class FeeSchedule:
    """
    Fee schedule for a symbol or exchange.

    Supports multiple fee structures:
    - Percentage: fee = notional * rate (e.g., crypto exchanges)
    - Per-share: fee = shares * rate (e.g., some stock brokers)
    - Flat: fixed fee per trade
    - Tiered: different rates based on volume

    Attributes:
        structure: Type of fee calculation
        maker_rate: Maker fee rate (bps for percentage, $ for per-share)
        taker_rate: Taker fee rate
        flat_fee: Flat fee per trade (if applicable)
        min_fee: Minimum fee per trade
        max_fee: Maximum fee per trade (cap)
        currency: Fee currency (usually quote currency)
        volume_tiers: Volume-based tier discounts {volume_threshold: rate_multiplier}
        rebate_enabled: Whether maker rebates are possible
    """

    structure: FeeStructure = FeeStructure.PERCENTAGE
    maker_rate: float = 10.0  # bps for percentage, $ for per-share
    taker_rate: float = 10.0
    flat_fee: float = 0.0
    min_fee: float = 0.0
    max_fee: Optional[float] = None
    currency: str = "USD"
    volume_tiers: Dict[float, float] = field(default_factory=dict)  # {volume: rate_multiplier}
    rebate_enabled: bool = False

    def compute_fee(
        self,
        notional: float,
        qty: float,
        is_maker: bool,
        monthly_volume: float = 0.0,
    ) -> float:
        """
        Compute fee for a trade.

        Args:
            notional: Trade value (price * qty)
            qty: Number of shares/units
            is_maker: True for maker orders
            monthly_volume: Monthly trading volume for tier calculation

        Returns:
            Fee amount in fee currency
        """
        base_rate = self.maker_rate if is_maker else self.taker_rate

        # Apply volume tier discount if applicable
        tier_mult = 1.0
        for threshold, mult in sorted(self.volume_tiers.items(), reverse=True):
            if monthly_volume >= threshold:
                tier_mult = mult
                break

        effective_rate = base_rate * tier_mult

        if self.structure == FeeStructure.PERCENTAGE:
            # Rate is in bps (basis points)
            fee = notional * (effective_rate / 10_000)
        elif self.structure == FeeStructure.PER_SHARE:
            # Rate is $ per share
            fee = abs(qty) * effective_rate
        elif self.structure == FeeStructure.FLAT:
            fee = self.flat_fee
        else:
            # Default to percentage
            fee = notional * (effective_rate / 10_000)

        # Add flat fee if mixed structure
        if self.structure == FeeStructure.MIXED:
            fee += self.flat_fee

        # Apply min/max constraints
        if self.min_fee > 0:
            fee = max(fee, self.min_fee)
        if self.max_fee is not None and self.max_fee > 0:
            fee = min(fee, self.max_fee)

        return fee

    def to_dict(self) -> Dict[str, Any]:
        return {
            "structure": self.structure.value,
            "maker_rate": self.maker_rate,
            "taker_rate": self.taker_rate,
            "flat_fee": self.flat_fee,
            "min_fee": self.min_fee,
            "max_fee": self.max_fee,
            "currency": self.currency,
            "volume_tiers": self.volume_tiers,
            "rebate_enabled": self.rebate_enabled,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FeeSchedule":
        structure_val = d.get("structure", "percentage")
        try:
            structure = FeeStructure(str(structure_val))
        except ValueError:
            structure = FeeStructure.PERCENTAGE

        return cls(
            structure=structure,
            maker_rate=float(d.get("maker_rate", 10.0)),
            taker_rate=float(d.get("taker_rate", 10.0)),
            flat_fee=float(d.get("flat_fee", 0.0)),
            min_fee=float(d.get("min_fee", 0.0)),
            max_fee=float(d.get("max_fee")) if d.get("max_fee") is not None else None,
            currency=str(d.get("currency", "USD")),
            volume_tiers={float(k): float(v) for k, v in d.get("volume_tiers", {}).items()},
            rebate_enabled=bool(d.get("rebate_enabled", False)),
        )


# =========================
# Account Info
# =========================


@dataclass(frozen=True)
class AccountInfo:
    """
    Exchange account information.

    Attributes:
        vendor: Exchange vendor
        account_id: Account identifier
        account_type: Account type (margin, cash, etc.)
        vip_tier: VIP/tier level for fee discounts
        maker_fee_rate: Effective maker fee rate
        taker_fee_rate: Effective taker fee rate
        buying_power: Available buying power
        cash_balance: Cash/USDT balance
        margin_enabled: Whether margin is enabled
        pattern_day_trader: PDT status (for US equities)
        raw_data: Original account data from exchange
    """

    vendor: ExchangeVendor
    account_id: str = ""
    account_type: str = "cash"
    vip_tier: int = 0
    maker_fee_rate: Optional[float] = None  # bps
    taker_fee_rate: Optional[float] = None  # bps
    buying_power: Optional[Decimal] = None
    cash_balance: Optional[Decimal] = None
    margin_enabled: bool = False
    pattern_day_trader: bool = False
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vendor": self.vendor.value,
            "account_id": self.account_id,
            "account_type": self.account_type,
            "vip_tier": self.vip_tier,
            "maker_fee_rate": self.maker_fee_rate,
            "taker_fee_rate": self.taker_fee_rate,
            "buying_power": str(self.buying_power) if self.buying_power else None,
            "cash_balance": str(self.cash_balance) if self.cash_balance else None,
            "margin_enabled": self.margin_enabled,
            "pattern_day_trader": self.pattern_day_trader,
            "raw_data": self.raw_data,
        }


# =========================
# Symbol Info
# =========================


@dataclass(frozen=True)
class SymbolInfo:
    """
    Comprehensive symbol information combining exchange rules and metadata.

    This is the primary model for symbol information across exchanges.
    It combines trading rules, asset details, and exchange-specific metadata.

    Attributes:
        symbol: Trading symbol
        vendor: Exchange vendor
        market_type: Type of market
        exchange_rule: Trading rules and constraints
        fee_schedule: Fee structure for this symbol
        name: Full name/description
        sector: Sector (for equities)
        industry: Industry (for equities)
        is_etf: Whether symbol is an ETF
        is_fractionable: Whether fractional shares are allowed
        status: Trading status
        listed_date: Listing date (ISO format)
        delisted_date: Delisting date if applicable
        raw_data: Original exchange data
    """

    symbol: str
    vendor: ExchangeVendor
    market_type: MarketType
    exchange_rule: ExchangeRule
    fee_schedule: Optional[FeeSchedule] = None
    name: str = ""
    sector: str = ""
    industry: str = ""
    is_etf: bool = False
    is_fractionable: bool = False
    status: str = "active"
    listed_date: Optional[str] = None
    delisted_date: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_tradable(self) -> bool:
        return self.exchange_rule.is_tradable and self.status == "active"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "vendor": self.vendor.value,
            "market_type": self.market_type.value,
            "exchange_rule": self.exchange_rule.to_dict(),
            "fee_schedule": self.fee_schedule.to_dict() if self.fee_schedule else None,
            "name": self.name,
            "sector": self.sector,
            "industry": self.industry,
            "is_etf": self.is_etf,
            "is_fractionable": self.is_fractionable,
            "status": self.status,
            "listed_date": self.listed_date,
            "delisted_date": self.delisted_date,
            "raw_data": self.raw_data,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SymbolInfo":
        vendor_val = d.get("vendor", "unknown")
        try:
            vendor = ExchangeVendor(str(vendor_val))
        except ValueError:
            vendor = ExchangeVendor.UNKNOWN

        market_type_val = d.get("market_type", "CRYPTO_SPOT")
        try:
            market_type = MarketType(str(market_type_val))
        except ValueError:
            market_type = MarketType.CRYPTO_SPOT

        rule_data = d.get("exchange_rule", {})
        if isinstance(rule_data, ExchangeRule):
            exchange_rule = rule_data
        else:
            exchange_rule = ExchangeRule.from_dict(rule_data)

        fee_data = d.get("fee_schedule")
        fee_schedule = FeeSchedule.from_dict(fee_data) if fee_data else None

        return cls(
            symbol=str(d.get("symbol", "")),
            vendor=vendor,
            market_type=market_type,
            exchange_rule=exchange_rule,
            fee_schedule=fee_schedule,
            name=str(d.get("name", "")),
            sector=str(d.get("sector", "")),
            industry=str(d.get("industry", "")),
            is_etf=bool(d.get("is_etf", False)),
            is_fractionable=bool(d.get("is_fractionable", False)),
            status=str(d.get("status", "active")),
            listed_date=d.get("listed_date"),
            delisted_date=d.get("delisted_date"),
            raw_data=dict(d.get("raw_data", {})),
        )


# =========================
# Predefined Sessions
# =========================

# US Equity Market Sessions (Eastern Time)
US_EQUITY_SESSIONS = [
    TradingSession(
        session_type=SessionType.PRE_MARKET,
        start_minutes=4 * 60,  # 04:00 ET
        end_minutes=9 * 60 + 30,  # 09:30 ET
        timezone="America/New_York",
        days_of_week=(0, 1, 2, 3, 4),
    ),
    TradingSession(
        session_type=SessionType.REGULAR,
        start_minutes=9 * 60 + 30,  # 09:30 ET
        end_minutes=16 * 60,  # 16:00 ET
        timezone="America/New_York",
        days_of_week=(0, 1, 2, 3, 4),
    ),
    TradingSession(
        session_type=SessionType.AFTER_HOURS,
        start_minutes=16 * 60,  # 16:00 ET
        end_minutes=20 * 60,  # 20:00 ET
        timezone="America/New_York",
        days_of_week=(0, 1, 2, 3, 4),
    ),
]

# Crypto 24/7 Session
CRYPTO_CONTINUOUS_SESSION = TradingSession(
    session_type=SessionType.CONTINUOUS,
    start_minutes=0,
    end_minutes=0,  # 0-0 means 24h
    timezone="UTC",
    days_of_week=(0, 1, 2, 3, 4, 5, 6),
)


# =========================
# Default Calendars
# =========================


def create_us_equity_calendar(vendor: ExchangeVendor = ExchangeVendor.ALPACA) -> MarketCalendar:
    """Create US equity market calendar."""
    return MarketCalendar(
        vendor=vendor,
        market_type=MarketType.EQUITY,
        sessions=list(US_EQUITY_SESSIONS),
        timezone="America/New_York",
    )


def create_crypto_calendar(vendor: ExchangeVendor = ExchangeVendor.BINANCE) -> MarketCalendar:
    """Create 24/7 crypto market calendar."""
    return MarketCalendar(
        vendor=vendor,
        market_type=MarketType.CRYPTO_SPOT,
        sessions=[CRYPTO_CONTINUOUS_SESSION],
        timezone="UTC",
    )


# =========================
# Corporate Actions (Phase 7)
# =========================


class CorporateActionType(str, Enum):
    """Type of corporate action affecting stock price/position."""

    DIVIDEND = "dividend"  # Cash dividend
    STOCK_DIVIDEND = "stock_dividend"  # Stock dividend (shares distributed)
    SPLIT = "split"  # Stock split (forward or reverse)
    MERGER = "merger"  # Merger/acquisition
    SPINOFF = "spinoff"  # Spinoff of subsidiary
    RIGHTS = "rights"  # Rights offering
    SYMBOL_CHANGE = "symbol_change"  # Ticker symbol change
    DELISTING = "delisting"  # Removed from exchange


class DividendType(str, Enum):
    """Type of dividend distribution."""

    REGULAR = "regular"  # Regular quarterly/monthly dividend
    SPECIAL = "special"  # One-time special dividend
    STOCK = "stock"  # Paid in shares, not cash
    QUALIFIED = "qualified"  # Tax-advantaged qualified dividend
    UNQUALIFIED = "unqualified"  # Ordinary income dividend


@dataclass(frozen=True)
class CorporateAction:
    """
    A corporate action affecting stock price or position.

    Corporate actions are critical for accurate backtesting:
    - Dividends: Affect total return, may need price adjustment
    - Splits: Multiply/divide shares, adjust prices proportionally
    - Mergers: May result in cash/stock consideration
    - Spinoffs: Create new position in spun-off company

    Attributes:
        action_type: Type of corporate action
        symbol: Stock symbol
        ex_date: Ex-dividend/ex-date (when price adjusts)
        record_date: Record date (ownership cutoff)
        pay_date: Payment/effective date
        announcement_date: When action was announced
        amount: Cash amount per share (dividends)
        ratio: Split ratio as tuple (new_shares, old_shares)
                e.g., (2, 1) for 2-for-1 split, (1, 10) for 1-for-10 reverse
        currency: Currency of cash amounts
        description: Human-readable description
        adjustment_factor: Price adjustment factor (e.g., 0.5 for 2:1 split)
        related_symbol: New symbol for mergers/spinoffs/symbol changes
        raw_data: Original data from source
    """

    action_type: CorporateActionType
    symbol: str
    ex_date: str  # ISO format YYYY-MM-DD
    record_date: Optional[str] = None
    pay_date: Optional[str] = None
    announcement_date: Optional[str] = None
    amount: Optional[Decimal] = None  # Per share amount for dividends
    ratio: Optional[Tuple[int, int]] = None  # (new, old) for splits
    currency: str = "USD"
    description: str = ""
    adjustment_factor: Optional[float] = None  # Pre-computed for convenience
    related_symbol: Optional[str] = None  # For mergers, spinoffs, symbol changes
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Compute adjustment factor if not provided."""
        if self.adjustment_factor is None and self.ratio is not None:
            # For split: new_shares / old_shares
            # e.g., 2:1 split → adjustment_factor = 0.5 (prices halve)
            new, old = self.ratio
            if old > 0:
                object.__setattr__(self, "adjustment_factor", old / new)

    @property
    def is_price_adjusting(self) -> bool:
        """Returns True if this action requires price adjustment."""
        return self.action_type in (
            CorporateActionType.DIVIDEND,
            CorporateActionType.STOCK_DIVIDEND,
            CorporateActionType.SPLIT,
            CorporateActionType.SPINOFF,
        )

    @property
    def is_position_adjusting(self) -> bool:
        """Returns True if this action changes position size."""
        return self.action_type in (
            CorporateActionType.SPLIT,
            CorporateActionType.STOCK_DIVIDEND,
            CorporateActionType.MERGER,
            CorporateActionType.SPINOFF,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "symbol": self.symbol,
            "ex_date": self.ex_date,
            "record_date": self.record_date,
            "pay_date": self.pay_date,
            "announcement_date": self.announcement_date,
            "amount": str(self.amount) if self.amount else None,
            "ratio": list(self.ratio) if self.ratio else None,
            "currency": self.currency,
            "description": self.description,
            "adjustment_factor": self.adjustment_factor,
            "related_symbol": self.related_symbol,
            "raw_data": self.raw_data,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CorporateAction":
        action_type_val = d.get("action_type", "dividend")
        try:
            action_type = CorporateActionType(str(action_type_val))
        except ValueError:
            action_type = CorporateActionType.DIVIDEND

        ratio = d.get("ratio")
        if ratio and isinstance(ratio, (list, tuple)) and len(ratio) >= 2:
            ratio = (int(ratio[0]), int(ratio[1]))
        else:
            ratio = None

        amount = d.get("amount")
        if amount is not None:
            try:
                amount = Decimal(str(amount))
            except Exception:
                amount = None

        return cls(
            action_type=action_type,
            symbol=str(d.get("symbol", "")),
            ex_date=str(d.get("ex_date", "")),
            record_date=d.get("record_date"),
            pay_date=d.get("pay_date"),
            announcement_date=d.get("announcement_date"),
            amount=amount,
            ratio=ratio,
            currency=str(d.get("currency", "USD")),
            description=str(d.get("description", "")),
            adjustment_factor=float(d["adjustment_factor"]) if d.get("adjustment_factor") else None,
            related_symbol=d.get("related_symbol"),
            raw_data=dict(d.get("raw_data", {})),
        )


@dataclass(frozen=True)
class Dividend:
    """
    Dividend payment details.

    Convenience class for dividend-specific operations.
    More detailed than CorporateAction for dividend analysis.

    Attributes:
        symbol: Stock symbol
        ex_date: Ex-dividend date
        record_date: Record date
        pay_date: Payment date
        declaration_date: Declaration/announcement date
        amount: Dividend amount per share
        dividend_type: Type of dividend (regular, special, etc.)
        frequency: Payment frequency (quarterly, monthly, annual)
        currency: Payment currency
        yield_pct: Dividend yield at declaration (optional)
        is_adjusted: Whether amount is split-adjusted
    """

    symbol: str
    ex_date: str  # ISO format
    amount: Decimal
    record_date: Optional[str] = None
    pay_date: Optional[str] = None
    declaration_date: Optional[str] = None
    dividend_type: DividendType = DividendType.REGULAR
    frequency: Optional[str] = None  # "quarterly", "monthly", "annual", etc.
    currency: str = "USD"
    yield_pct: Optional[float] = None
    is_adjusted: bool = False  # True if amount is split-adjusted

    @property
    def ex_date_timestamp(self) -> int:
        """Ex-date as Unix timestamp (seconds)."""
        from datetime import datetime

        dt = datetime.fromisoformat(self.ex_date)
        return int(dt.timestamp())

    def to_corporate_action(self) -> CorporateAction:
        """Convert to CorporateAction for unified processing."""
        return CorporateAction(
            action_type=CorporateActionType.DIVIDEND,
            symbol=self.symbol,
            ex_date=self.ex_date,
            record_date=self.record_date,
            pay_date=self.pay_date,
            announcement_date=self.declaration_date,
            amount=self.amount,
            currency=self.currency,
            description=f"{self.dividend_type.value} dividend: ${self.amount}",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "ex_date": self.ex_date,
            "amount": str(self.amount),
            "record_date": self.record_date,
            "pay_date": self.pay_date,
            "declaration_date": self.declaration_date,
            "dividend_type": self.dividend_type.value,
            "frequency": self.frequency,
            "currency": self.currency,
            "yield_pct": self.yield_pct,
            "is_adjusted": self.is_adjusted,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Dividend":
        div_type_val = d.get("dividend_type", "regular")
        try:
            div_type = DividendType(str(div_type_val))
        except ValueError:
            div_type = DividendType.REGULAR

        amount = d.get("amount", "0")
        try:
            amount = Decimal(str(amount))
        except Exception:
            amount = Decimal("0")

        return cls(
            symbol=str(d.get("symbol", "")),
            ex_date=str(d.get("ex_date", "")),
            amount=amount,
            record_date=d.get("record_date"),
            pay_date=d.get("pay_date"),
            declaration_date=d.get("declaration_date"),
            dividend_type=div_type,
            frequency=d.get("frequency"),
            currency=str(d.get("currency", "USD")),
            yield_pct=float(d["yield_pct"]) if d.get("yield_pct") else None,
            is_adjusted=bool(d.get("is_adjusted", False)),
        )


@dataclass(frozen=True)
class StockSplit:
    """
    Stock split details.

    Attributes:
        symbol: Stock symbol
        ex_date: Ex-date (effective date)
        ratio: Split ratio as (new_shares, old_shares)
               e.g., (4, 1) means 4-for-1 split
               (1, 10) means 1-for-10 reverse split
        announcement_date: When split was announced
        is_reverse: True if reverse split (consolidation)
    """

    symbol: str
    ex_date: str  # ISO format
    ratio: Tuple[int, int]  # (new, old)
    announcement_date: Optional[str] = None
    is_reverse: bool = False

    def __post_init__(self) -> None:
        """Validate and set is_reverse flag."""
        new, old = self.ratio
        if new < old and not self.is_reverse:
            object.__setattr__(self, "is_reverse", True)

    @property
    def adjustment_factor(self) -> float:
        """
        Factor to multiply historical prices by.

        For 2:1 split, factor = 0.5 (prices halve)
        For 1:10 reverse, factor = 10 (prices 10x)
        """
        new, old = self.ratio
        return old / new if new > 0 else 1.0

    @property
    def share_multiplier(self) -> float:
        """
        Factor to multiply share count by.

        For 2:1 split, multiplier = 2 (shares double)
        For 1:10 reverse, multiplier = 0.1 (shares reduced)
        """
        new, old = self.ratio
        return new / old if old > 0 else 1.0

    def to_corporate_action(self) -> CorporateAction:
        """Convert to CorporateAction for unified processing."""
        split_type = "reverse split" if self.is_reverse else "split"
        return CorporateAction(
            action_type=CorporateActionType.SPLIT,
            symbol=self.symbol,
            ex_date=self.ex_date,
            announcement_date=self.announcement_date,
            ratio=self.ratio,
            adjustment_factor=self.adjustment_factor,
            description=f"{self.ratio[0]}-for-{self.ratio[1]} {split_type}",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "ex_date": self.ex_date,
            "ratio": list(self.ratio),
            "announcement_date": self.announcement_date,
            "is_reverse": self.is_reverse,
            "adjustment_factor": self.adjustment_factor,
            "share_multiplier": self.share_multiplier,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "StockSplit":
        ratio = d.get("ratio", [1, 1])
        if isinstance(ratio, (list, tuple)) and len(ratio) >= 2:
            ratio = (int(ratio[0]), int(ratio[1]))
        else:
            ratio = (1, 1)

        return cls(
            symbol=str(d.get("symbol", "")),
            ex_date=str(d.get("ex_date", "")),
            ratio=ratio,
            announcement_date=d.get("announcement_date"),
            is_reverse=bool(d.get("is_reverse", False)),
        )


@dataclass(frozen=True)
class EarningsEvent:
    """
    Earnings announcement event.

    Important for:
    - Volatility modeling (earnings surprises)
    - Position sizing around earnings
    - Feature engineering (days to/from earnings)

    Attributes:
        symbol: Stock symbol
        report_date: Expected/actual report date
        fiscal_quarter: Fiscal quarter (1-4)
        fiscal_year: Fiscal year
        report_time: "BMO" (before market open), "AMC" (after market close), or "DDM" (during day)
        eps_estimate: Consensus EPS estimate
        eps_actual: Actual reported EPS (None if not yet reported)
        revenue_estimate: Consensus revenue estimate
        revenue_actual: Actual revenue
        surprise_pct: EPS surprise percentage (actual - estimate) / estimate
        is_confirmed: Whether date is confirmed by company
    """

    symbol: str
    report_date: str  # ISO format YYYY-MM-DD
    fiscal_quarter: Optional[int] = None  # 1-4
    fiscal_year: Optional[int] = None
    report_time: Optional[str] = None  # "BMO", "AMC", "DDM"
    eps_estimate: Optional[Decimal] = None
    eps_actual: Optional[Decimal] = None
    revenue_estimate: Optional[Decimal] = None  # In millions or actual
    revenue_actual: Optional[Decimal] = None
    surprise_pct: Optional[float] = None
    is_confirmed: bool = False

    @property
    def report_date_timestamp(self) -> int:
        """Report date as Unix timestamp (seconds)."""
        from datetime import datetime

        dt = datetime.fromisoformat(self.report_date)
        return int(dt.timestamp())

    @property
    def has_reported(self) -> bool:
        """Whether earnings have been reported."""
        return self.eps_actual is not None

    @property
    def beat_estimates(self) -> Optional[bool]:
        """Whether company beat EPS estimates (None if not reported)."""
        if self.eps_actual is None or self.eps_estimate is None:
            return None
        return self.eps_actual > self.eps_estimate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "report_date": self.report_date,
            "fiscal_quarter": self.fiscal_quarter,
            "fiscal_year": self.fiscal_year,
            "report_time": self.report_time,
            "eps_estimate": str(self.eps_estimate) if self.eps_estimate else None,
            "eps_actual": str(self.eps_actual) if self.eps_actual else None,
            "revenue_estimate": str(self.revenue_estimate) if self.revenue_estimate else None,
            "revenue_actual": str(self.revenue_actual) if self.revenue_actual else None,
            "surprise_pct": self.surprise_pct,
            "is_confirmed": self.is_confirmed,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EarningsEvent":
        def _to_decimal_opt(v: Any) -> Optional[Decimal]:
            if v is None:
                return None
            try:
                return Decimal(str(v))
            except Exception:
                return None

        return cls(
            symbol=str(d.get("symbol", "")),
            report_date=str(d.get("report_date", "")),
            fiscal_quarter=int(d["fiscal_quarter"]) if d.get("fiscal_quarter") else None,
            fiscal_year=int(d["fiscal_year"]) if d.get("fiscal_year") else None,
            report_time=d.get("report_time"),
            eps_estimate=_to_decimal_opt(d.get("eps_estimate")),
            eps_actual=_to_decimal_opt(d.get("eps_actual")),
            revenue_estimate=_to_decimal_opt(d.get("revenue_estimate")),
            revenue_actual=_to_decimal_opt(d.get("revenue_actual")),
            surprise_pct=float(d["surprise_pct"]) if d.get("surprise_pct") else None,
            is_confirmed=bool(d.get("is_confirmed", False)),
        )


@dataclass(frozen=True)
class AdjustmentFactors:
    """
    Combined adjustment factors for a symbol at a point in time.

    Used for converting between split-adjusted and unadjusted prices.

    Attributes:
        symbol: Stock symbol
        date: Date these factors apply from
        split_factor: Cumulative split adjustment factor
        dividend_factor: Cumulative dividend adjustment factor
        combined_factor: Product of split and dividend factors
    """

    symbol: str
    date: str  # ISO format
    split_factor: float = 1.0
    dividend_factor: float = 1.0

    @property
    def combined_factor(self) -> float:
        """Total adjustment factor (split * dividend)."""
        return self.split_factor * self.dividend_factor

    def adjust_price(self, price: float) -> float:
        """Apply adjustment to convert raw price to adjusted price."""
        return price * self.combined_factor

    def unadjust_price(self, adjusted_price: float) -> float:
        """Reverse adjustment to get raw price from adjusted price."""
        if self.combined_factor == 0:
            return adjusted_price
        return adjusted_price / self.combined_factor

    def adjust_volume(self, volume: float) -> float:
        """Adjust volume (inverse of price adjustment for splits)."""
        if self.split_factor == 0:
            return volume
        return volume / self.split_factor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "date": self.date,
            "split_factor": self.split_factor,
            "dividend_factor": self.dividend_factor,
            "combined_factor": self.combined_factor,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AdjustmentFactors":
        return cls(
            symbol=str(d.get("symbol", "")),
            date=str(d.get("date", "")),
            split_factor=float(d.get("split_factor", 1.0)),
            dividend_factor=float(d.get("dividend_factor", 1.0)),
        )


# =========================
# Forex Models (Phase 0)
# =========================


class ForexSessionType(str, Enum):
    """
    Forex trading session type.

    The forex market operates in overlapping sessions with varying liquidity:
    - Sydney: Lowest liquidity, widest spreads
    - Tokyo: Moderate liquidity, JPY pairs active
    - London: Highest liquidity, EUR/GBP pairs active
    - New York: High liquidity, USD pairs active
    - Overlaps: Best liquidity and tightest spreads

    References:
        - BIS Triennial Survey 2022: https://www.bis.org/statistics/rpfx22.htm
        - Forex session times: Standard GMT/UTC-based windows
    """

    SYDNEY = "sydney"
    TOKYO = "tokyo"
    LONDON = "london"
    NEW_YORK = "new_york"
    LONDON_NY_OVERLAP = "london_ny_overlap"
    TOKYO_LONDON_OVERLAP = "tokyo_london_overlap"
    WEEKEND = "weekend"
    OFF_HOURS = "off_hours"  # Between sessions, low liquidity


class CurrencyPairCategory(str, Enum):
    """
    Currency pair classification by liquidity and spread characteristics.

    Categories based on BIS Triennial Survey 2022 volume data:
    - MAJOR: USD pairs with G7 currencies, $500B+ daily volume
    - MINOR: Cross rates without USD but major currencies, $50-100B
    - CROSS: JPY crosses and other liquid crosses, $20-50B
    - EXOTIC: Emerging market currencies, $1-10B

    References:
        - BIS Triennial Survey 2022
        - Standard institutional classification
    """

    MAJOR = "major"  # EUR/USD, USD/JPY, GBP/USD, USD/CHF, AUD/USD, USD/CAD, NZD/USD
    MINOR = "minor"  # EUR/GBP, EUR/CHF, GBP/CHF, EUR/AUD, GBP/AUD
    CROSS = "cross"  # EUR/JPY, GBP/JPY, AUD/JPY, CHF/JPY, CAD/JPY, NZD/JPY
    EXOTIC = "exotic"  # USD/TRY, USD/ZAR, USD/MXN, USD/SGD, USD/HKD, USD/NOK, USD/SEK


@dataclass(frozen=True)
class ForexSessionWindow:
    """
    Forex session trading window with liquidity characteristics.

    Attributes:
        session_type: Type of forex session
        start_hour_utc: Start hour in UTC (0-23)
        end_hour_utc: End hour in UTC (0-23), can be < start for overnight
        liquidity_factor: Relative liquidity (1.0 = London session baseline)
        spread_multiplier: Spread adjustment (1.0 = tightest spreads, >1 = wider)
        days_of_week: Active days (0=Monday, 6=Sunday)

    Note:
        liquidity_factor and spread_multiplier are inversely related:
        Higher liquidity → Lower spread multiplier
    """

    session_type: ForexSessionType
    start_hour_utc: int
    end_hour_utc: int
    liquidity_factor: float
    spread_multiplier: float
    days_of_week: Tuple[int, ...] = (0, 1, 2, 3, 4)  # Mon-Fri by default

    def __post_init__(self) -> None:
        """Validate session window parameters."""
        if not (0 <= self.start_hour_utc <= 23):
            raise ValueError(f"start_hour_utc must be 0-23, got {self.start_hour_utc}")
        if not (0 <= self.end_hour_utc <= 23):
            raise ValueError(f"end_hour_utc must be 0-23, got {self.end_hour_utc}")
        # Allow 0.0 for weekend (no trading) but reject negative
        if self.liquidity_factor < 0:
            raise ValueError(f"liquidity_factor must be non-negative, got {self.liquidity_factor}")
        # Allow infinity for weekend spread but reject zero/negative
        if self.spread_multiplier <= 0 and not math.isinf(self.spread_multiplier):
            raise ValueError(f"spread_multiplier must be positive, got {self.spread_multiplier}")

    @property
    def is_overnight(self) -> bool:
        """Returns True if session spans midnight UTC."""
        return self.end_hour_utc < self.start_hour_utc

    @property
    def duration_hours(self) -> int:
        """Duration of session in hours."""
        if self.is_overnight:
            return (24 - self.start_hour_utc) + self.end_hour_utc
        return self.end_hour_utc - self.start_hour_utc

    def contains_hour(self, hour_utc: int, day_of_week: int) -> bool:
        """
        Check if given UTC hour is within this session.

        Args:
            hour_utc: Hour in UTC (0-23)
            day_of_week: Day of week (0=Monday, 6=Sunday)

        Returns:
            True if hour is within session window
        """
        if day_of_week not in self.days_of_week:
            return False

        if self.is_overnight:
            return hour_utc >= self.start_hour_utc or hour_utc < self.end_hour_utc
        return self.start_hour_utc <= hour_utc < self.end_hour_utc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_type": self.session_type.value,
            "start_hour_utc": self.start_hour_utc,
            "end_hour_utc": self.end_hour_utc,
            "liquidity_factor": self.liquidity_factor,
            "spread_multiplier": self.spread_multiplier,
            "days_of_week": list(self.days_of_week),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ForexSessionWindow":
        session_type_val = d.get("session_type", "off_hours")
        try:
            session_type = ForexSessionType(str(session_type_val))
        except ValueError:
            session_type = ForexSessionType.OFF_HOURS

        days = d.get("days_of_week", [0, 1, 2, 3, 4])
        if isinstance(days, (list, tuple)):
            days_tuple = tuple(int(x) for x in days)
        else:
            days_tuple = (0, 1, 2, 3, 4)

        return cls(
            session_type=session_type,
            start_hour_utc=int(d.get("start_hour_utc", 0)),
            end_hour_utc=int(d.get("end_hour_utc", 0)),
            liquidity_factor=float(d.get("liquidity_factor", 1.0)),
            spread_multiplier=float(d.get("spread_multiplier", 1.0)),
            days_of_week=days_tuple,
        )


# =========================
# Forex Session Constants
# =========================

# Session windows based on standard forex market hours
# Liquidity factors relative to London session (peak = 1.0)
# Spread multipliers: 1.0 = tightest (London/NY overlap), >1 = wider

FOREX_SESSION_WINDOWS: List[ForexSessionWindow] = [
    # Sydney Session: 21:00-06:00 UTC (lowest liquidity for major pairs)
    ForexSessionWindow(
        session_type=ForexSessionType.SYDNEY,
        start_hour_utc=21,
        end_hour_utc=6,
        liquidity_factor=0.65,
        spread_multiplier=1.5,
        days_of_week=(0, 1, 2, 3, 4, 6),  # Sun evening to Fri, starts Sun
    ),
    # Tokyo Session: 00:00-09:00 UTC (good for JPY pairs)
    ForexSessionWindow(
        session_type=ForexSessionType.TOKYO,
        start_hour_utc=0,
        end_hour_utc=9,
        liquidity_factor=0.75,
        spread_multiplier=1.3,
        days_of_week=(0, 1, 2, 3, 4),  # Mon-Fri
    ),
    # London Session: 07:00-16:00 UTC (highest overall liquidity)
    ForexSessionWindow(
        session_type=ForexSessionType.LONDON,
        start_hour_utc=7,
        end_hour_utc=16,
        liquidity_factor=1.10,
        spread_multiplier=1.0,
        days_of_week=(0, 1, 2, 3, 4),  # Mon-Fri
    ),
    # New York Session: 12:00-21:00 UTC (high USD pair liquidity)
    ForexSessionWindow(
        session_type=ForexSessionType.NEW_YORK,
        start_hour_utc=12,
        end_hour_utc=21,
        liquidity_factor=1.05,
        spread_multiplier=1.0,
        days_of_week=(0, 1, 2, 3, 4),  # Mon-Fri
    ),
    # London/NY Overlap: 12:00-16:00 UTC (BEST liquidity, tightest spreads)
    ForexSessionWindow(
        session_type=ForexSessionType.LONDON_NY_OVERLAP,
        start_hour_utc=12,
        end_hour_utc=16,
        liquidity_factor=1.35,
        spread_multiplier=0.8,  # Tightest spreads!
        days_of_week=(0, 1, 2, 3, 4),  # Mon-Fri
    ),
    # Tokyo/London Overlap: 07:00-09:00 UTC (moderate boost)
    ForexSessionWindow(
        session_type=ForexSessionType.TOKYO_LONDON_OVERLAP,
        start_hour_utc=7,
        end_hour_utc=9,
        liquidity_factor=0.90,
        spread_multiplier=1.1,
        days_of_week=(0, 1, 2, 3, 4),  # Mon-Fri
    ),
]

# Weekend: Forex market closed from Fri 21:00 UTC to Sun 21:00 UTC
FOREX_WEEKEND_WINDOW = ForexSessionWindow(
    session_type=ForexSessionType.WEEKEND,
    start_hour_utc=21,  # Friday 21:00 UTC
    end_hour_utc=21,  # Sunday 21:00 UTC (special case: 48h)
    liquidity_factor=0.0,  # No trading
    spread_multiplier=float("inf"),  # Cannot trade
    days_of_week=(4, 5, 6),  # Fri evening, Sat, Sun until evening
)


# =========================
# Forex Pip Sizes
# =========================

# Standard pip sizes by base currency
# Most pairs: 0.0001 (4 decimal places)
# JPY pairs: 0.01 (2 decimal places)

PIP_SIZE_BY_QUOTE_CURRENCY: Dict[str, float] = {
    "JPY": 0.01,  # Japanese Yen pairs
    "HUF": 0.01,  # Hungarian Forint
    "default": 0.0001,  # Standard 4-decimal pairs
}


def get_pip_size(symbol: str) -> float:
    """
    Get pip size for a currency pair.

    Args:
        symbol: Currency pair (e.g., "EUR_USD", "USD_JPY", "EUR/USD")

    Returns:
        Pip size (0.0001 for most, 0.01 for JPY pairs)

    Examples:
        >>> get_pip_size("EUR_USD")
        0.0001
        >>> get_pip_size("USD_JPY")
        0.01
        >>> get_pip_size("EUR/JPY")
        0.01
    """
    # Normalize symbol: EUR_USD -> EUR/USD -> check quote (second currency)
    normalized = symbol.upper().replace("_", "/")
    parts = normalized.split("/")
    if len(parts) == 2:
        quote_currency = parts[1]
        return PIP_SIZE_BY_QUOTE_CURRENCY.get(quote_currency, PIP_SIZE_BY_QUOTE_CURRENCY["default"])
    return PIP_SIZE_BY_QUOTE_CURRENCY["default"]


def pips_to_price(pips: float, symbol: str) -> float:
    """
    Convert pips to price difference.

    Args:
        pips: Number of pips
        symbol: Currency pair

    Returns:
        Price difference

    Examples:
        >>> pips_to_price(10, "EUR_USD")
        0.001
        >>> pips_to_price(10, "USD_JPY")
        0.1
    """
    return pips * get_pip_size(symbol)


def price_to_pips(price_diff: float, symbol: str) -> float:
    """
    Convert price difference to pips.

    Args:
        price_diff: Price difference
        symbol: Currency pair

    Returns:
        Number of pips

    Examples:
        >>> price_to_pips(0.001, "EUR_USD")
        10.0
        >>> price_to_pips(0.1, "USD_JPY")
        10.0
    """
    pip_size = get_pip_size(symbol)
    if pip_size == 0:
        return 0.0
    return price_diff / pip_size


# =========================
# Forex Pair Classification
# =========================

# Major pairs: USD with G7 currencies
FOREX_MAJOR_PAIRS: Tuple[str, ...] = (
    "EUR_USD",
    "USD_JPY",
    "GBP_USD",
    "USD_CHF",
    "AUD_USD",
    "USD_CAD",
    "NZD_USD",
)

# Minor pairs (crosses): G7 without USD
FOREX_MINOR_PAIRS: Tuple[str, ...] = (
    "EUR_GBP",
    "EUR_CHF",
    "GBP_CHF",
    "EUR_AUD",
    "GBP_AUD",
    "EUR_CAD",
    "GBP_CAD",
    "AUD_NZD",
    "AUD_CAD",
    "NZD_CAD",
)

# JPY crosses
FOREX_JPY_CROSSES: Tuple[str, ...] = (
    "EUR_JPY",
    "GBP_JPY",
    "AUD_JPY",
    "CHF_JPY",
    "CAD_JPY",
    "NZD_JPY",
)

# Exotic pairs (emerging markets)
FOREX_EXOTIC_PAIRS: Tuple[str, ...] = (
    "USD_TRY",
    "USD_ZAR",
    "USD_MXN",
    "USD_SGD",
    "USD_HKD",
    "USD_NOK",
    "USD_SEK",
    "USD_DKK",
    "USD_PLN",
    "USD_CZK",
    "USD_HUF",
    "USD_RUB",
    "EUR_TRY",
    "EUR_ZAR",
    "EUR_NOK",
    "EUR_SEK",
)


def classify_currency_pair(symbol: str) -> CurrencyPairCategory:
    """
    Classify a currency pair by its category.

    Args:
        symbol: Currency pair (e.g., "EUR_USD", "EUR/USD")

    Returns:
        CurrencyPairCategory enum value

    Examples:
        >>> classify_currency_pair("EUR_USD")
        CurrencyPairCategory.MAJOR
        >>> classify_currency_pair("EUR/JPY")
        CurrencyPairCategory.CROSS
        >>> classify_currency_pair("USD_TRY")
        CurrencyPairCategory.EXOTIC
    """
    normalized = symbol.upper().replace("/", "_")

    if normalized in FOREX_MAJOR_PAIRS:
        return CurrencyPairCategory.MAJOR
    elif normalized in FOREX_MINOR_PAIRS:
        return CurrencyPairCategory.MINOR
    elif normalized in FOREX_JPY_CROSSES:
        return CurrencyPairCategory.CROSS
    elif normalized in FOREX_EXOTIC_PAIRS:
        return CurrencyPairCategory.EXOTIC

    # Heuristic for unknown pairs
    parts = normalized.split("_")
    if len(parts) == 2:
        base, quote = parts
        major_currencies = {"EUR", "USD", "JPY", "GBP", "CHF", "AUD", "CAD", "NZD"}
        if base in major_currencies and quote in major_currencies:
            if "USD" in (base, quote):
                return CurrencyPairCategory.MAJOR
            elif "JPY" in (base, quote):
                return CurrencyPairCategory.CROSS
            return CurrencyPairCategory.MINOR
    return CurrencyPairCategory.EXOTIC


# =========================
# Forex Spread Profiles
# =========================


@dataclass(frozen=True)
class ForexSpreadProfile:
    """
    Spread profile for a currency pair category.

    Attributes:
        category: Currency pair category
        retail_spread_pips: Typical retail broker spread
        institutional_spread_pips: Institutional/ECN spread
        conservative_spread_pips: Conservative estimate for simulation
        avg_daily_range_pips: Average daily price range in pips

    References:
        - BIS Triennial Survey 2022
        - Major broker spread sheets
    """

    category: CurrencyPairCategory
    retail_spread_pips: float
    institutional_spread_pips: float
    conservative_spread_pips: float
    avg_daily_range_pips: float

    def get_spread(self, profile: str = "conservative") -> float:
        """Get spread in pips for given profile."""
        if profile == "retail":
            return self.retail_spread_pips
        elif profile == "institutional":
            return self.institutional_spread_pips
        return self.conservative_spread_pips


# Default spread profiles by category
FOREX_SPREAD_PROFILES: Dict[CurrencyPairCategory, ForexSpreadProfile] = {
    CurrencyPairCategory.MAJOR: ForexSpreadProfile(
        category=CurrencyPairCategory.MAJOR,
        retail_spread_pips=1.2,
        institutional_spread_pips=0.3,
        conservative_spread_pips=1.5,
        avg_daily_range_pips=75.0,
    ),
    CurrencyPairCategory.MINOR: ForexSpreadProfile(
        category=CurrencyPairCategory.MINOR,
        retail_spread_pips=2.0,
        institutional_spread_pips=0.6,
        conservative_spread_pips=2.5,
        avg_daily_range_pips=60.0,
    ),
    CurrencyPairCategory.CROSS: ForexSpreadProfile(
        category=CurrencyPairCategory.CROSS,
        retail_spread_pips=3.0,
        institutional_spread_pips=1.0,
        conservative_spread_pips=4.0,
        avg_daily_range_pips=100.0,
    ),
    CurrencyPairCategory.EXOTIC: ForexSpreadProfile(
        category=CurrencyPairCategory.EXOTIC,
        retail_spread_pips=30.0,
        institutional_spread_pips=10.0,
        conservative_spread_pips=50.0,
        avg_daily_range_pips=300.0,
    ),
}


def get_spread_profile(symbol: str) -> ForexSpreadProfile:
    """
    Get spread profile for a currency pair.

    Args:
        symbol: Currency pair (e.g., "EUR_USD")

    Returns:
        ForexSpreadProfile for the pair's category
    """
    category = classify_currency_pair(symbol)
    return FOREX_SPREAD_PROFILES[category]


# =========================
# Forex Calendar Helper
# =========================


def create_forex_calendar(vendor: ExchangeVendor = ExchangeVendor.OANDA) -> MarketCalendar:
    """
    Create forex market calendar.

    Forex market hours: Sunday 21:00 UTC to Friday 21:00 UTC
    (or Sunday 5pm ET to Friday 5pm ET)
    """
    # Create a continuous session for forex (with weekend closure handled separately)
    forex_session = TradingSession(
        session_type=SessionType.CONTINUOUS,
        start_minutes=21 * 60,  # Sunday 21:00 UTC
        end_minutes=21 * 60,  # Friday 21:00 UTC (special handling)
        timezone="UTC",
        days_of_week=(0, 1, 2, 3, 4),  # Active Mon-Fri
        is_active=True,
    )
    return MarketCalendar(
        vendor=vendor,
        market_type=MarketType.FOREX,
        sessions=[forex_session],
        timezone="UTC",
    )


def get_current_forex_session(hour_utc: int, day_of_week: int) -> ForexSessionType:
    """
    Determine current forex session based on UTC hour and day.

    Priority order for overlapping sessions:
    1. LONDON_NY_OVERLAP (best liquidity)
    2. TOKYO_LONDON_OVERLAP
    3. Individual sessions (LONDON > NEW_YORK > TOKYO > SYDNEY)
    4. OFF_HOURS

    Args:
        hour_utc: Current hour in UTC (0-23)
        day_of_week: Day of week (0=Monday, 6=Sunday)

    Returns:
        ForexSessionType for current session
    """
    # Check weekend first
    if day_of_week == 5:  # Saturday
        return ForexSessionType.WEEKEND
    if day_of_week == 6 and hour_utc < 21:  # Sunday before 21:00 UTC
        return ForexSessionType.WEEKEND
    if day_of_week == 4 and hour_utc >= 21:  # Friday after 21:00 UTC
        return ForexSessionType.WEEKEND

    # Check overlaps first (highest priority)
    for window in FOREX_SESSION_WINDOWS:
        if window.session_type == ForexSessionType.LONDON_NY_OVERLAP:
            if window.contains_hour(hour_utc, day_of_week):
                return ForexSessionType.LONDON_NY_OVERLAP

    for window in FOREX_SESSION_WINDOWS:
        if window.session_type == ForexSessionType.TOKYO_LONDON_OVERLAP:
            if window.contains_hour(hour_utc, day_of_week):
                return ForexSessionType.TOKYO_LONDON_OVERLAP

    # Check individual sessions in priority order
    session_priority = [
        ForexSessionType.LONDON,
        ForexSessionType.NEW_YORK,
        ForexSessionType.TOKYO,
        ForexSessionType.SYDNEY,
    ]

    for session_type in session_priority:
        for window in FOREX_SESSION_WINDOWS:
            if window.session_type == session_type:
                if window.contains_hour(hour_utc, day_of_week):
                    return session_type

    return ForexSessionType.OFF_HOURS


def get_session_liquidity_factor(hour_utc: int, day_of_week: int) -> float:
    """
    Get liquidity factor for current forex session.

    Args:
        hour_utc: Current hour in UTC (0-23)
        day_of_week: Day of week (0=Monday, 6=Sunday)

    Returns:
        Liquidity factor (1.0 = London baseline, 0.0 = weekend/closed)
    """
    session = get_current_forex_session(hour_utc, day_of_week)

    if session == ForexSessionType.WEEKEND:
        return 0.0

    for window in FOREX_SESSION_WINDOWS:
        if window.session_type == session:
            return window.liquidity_factor

    return 0.5  # Default for OFF_HOURS


def get_session_spread_multiplier(hour_utc: int, day_of_week: int) -> float:
    """
    Get spread multiplier for current forex session.

    Args:
        hour_utc: Current hour in UTC (0-23)
        day_of_week: Day of week (0=Monday, 6=Sunday)

    Returns:
        Spread multiplier (1.0 = London, <1 = London/NY overlap, >1 = low liquidity)
    """
    session = get_current_forex_session(hour_utc, day_of_week)

    if session == ForexSessionType.WEEKEND:
        return float("inf")  # Cannot trade

    for window in FOREX_SESSION_WINDOWS:
        if window.session_type == session:
            return window.spread_multiplier

    return 2.0  # Default for OFF_HOURS (wider spreads)
