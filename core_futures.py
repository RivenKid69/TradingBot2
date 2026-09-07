# -*- coding: utf-8 -*-
"""
core_futures.py
Unified futures trading models for ALL futures types.

Supports:
- Crypto Perpetual (Binance USDT-M, Bybit)
- Crypto Quarterly (Binance delivery)
- Index Futures (CME: ES, NQ, YM, RTY)
- Commodity Futures (COMEX: GC, SI; NYMEX: CL, NG)
- Currency Futures (CME: 6E, 6J, 6B, 6A)
- Bond Futures (CBOT: ZB, ZN, ZF)

Design Principles:
- Immutable dataclasses for thread safety and clarity
- Decimal for financial precision (prices, quantities)
- int milliseconds UTC for timestamps
- Backward compatible with existing core_models.py
- Vendor-agnostic models (Binance/CME differences handled in adapters)

References:
- Binance Futures API: https://binance-docs.github.io/apidocs/futures/en/
- CME Group Contract Specs: https://www.cmegroup.com/trading/equity-index/
- Almgren & Chriss (2001): Optimal Execution of Portfolio Transactions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple


# ============================================================================
# ENUMS (Unified across all futures types)
# ============================================================================


class FuturesType(str, Enum):
    """
    Unified futures type classification.

    Covers both crypto (Binance) and traditional (CME) futures.
    """

    # Crypto futures (Binance, Bybit, etc.)
    CRYPTO_PERPETUAL = "CRYPTO_PERPETUAL"  # No expiration, funding rate
    CRYPTO_QUARTERLY = "CRYPTO_QUARTERLY"  # Quarterly expiry (BTCUSDT_240329)

    # CME/CBOT/COMEX/NYMEX traditional futures
    INDEX_FUTURES = "INDEX_FUTURES"  # ES, NQ, YM, RTY
    COMMODITY_FUTURES = "COMMODITY_FUTURES"  # GC, CL, SI, NG
    CURRENCY_FUTURES = "CURRENCY_FUTURES"  # 6E, 6J, 6B, 6A
    BOND_FUTURES = "BOND_FUTURES"  # ZB, ZN, ZF (Treasury)

    @property
    def is_crypto(self) -> bool:
        """Check if this is a crypto futures type."""
        return self in (FuturesType.CRYPTO_PERPETUAL, FuturesType.CRYPTO_QUARTERLY)

    @property
    def is_perpetual(self) -> bool:
        """Check if this is a perpetual contract (no expiry)."""
        return self == FuturesType.CRYPTO_PERPETUAL

    @property
    def has_funding(self) -> bool:
        """Check if this type uses funding rate mechanism."""
        return self == FuturesType.CRYPTO_PERPETUAL

    @property
    def has_expiry(self) -> bool:
        """Check if this type has contract expiration."""
        return self != FuturesType.CRYPTO_PERPETUAL


class ContractType(str, Enum):
    """
    Contract expiration type.

    Defines the contract's relationship to expiration dates.
    """

    PERPETUAL = "PERPETUAL"  # No expiration (crypto perpetual)
    CURRENT_MONTH = "CURRENT_MONTH"  # Monthly contracts (CL, NG)
    CURRENT_QUARTER = "CURRENT_QUARTER"  # Front quarter (ES, NQ)
    NEXT_QUARTER = "NEXT_QUARTER"  # Next quarter
    BACK_MONTH = "BACK_MONTH"  # Further out months
    CONTINUOUS = "CONTINUOUS"  # Auto-rolling continuous contract


class SettlementType(str, Enum):
    """
    Settlement method at contract expiration or funding interval.

    Different settlement types have different P&L realization mechanics.
    """

    CASH = "CASH"  # Cash settlement (ES, NQ, 6E)
    PHYSICAL = "PHYSICAL"  # Physical delivery (GC, CL - most roll before)
    FUNDING = "FUNDING"  # Funding rate payments (crypto perpetual)


class MarginMode(str, Enum):
    """
    Margin mode for position management.

    Affects how margin is calculated and shared between positions.
    """

    CROSS = "CROSS"  # Shared margin across all positions (Binance cross, IB portfolio)
    ISOLATED = "ISOLATED"  # Per-position margin (Binance isolated)
    SPAN = "SPAN"  # CME SPAN margin (portfolio-based with offsets)


class PositionSide(str, Enum):
    """
    Position side for hedge mode trading.

    BOTH is for one-way mode (net position), LONG/SHORT for hedge mode.
    """

    BOTH = "BOTH"  # One-way mode (net position)
    LONG = "LONG"  # Hedge mode long position
    SHORT = "SHORT"  # Hedge mode short position


class Exchange(str, Enum):
    """
    Exchange where contract trades.

    Used for routing and exchange-specific logic.
    """

    BINANCE = "BINANCE"  # Binance Futures
    BYBIT = "BYBIT"  # Bybit Derivatives
    CME = "CME"  # E-mini indices (ES, NQ)
    COMEX = "COMEX"  # Gold, Silver
    NYMEX = "NYMEX"  # Oil, Natural Gas
    CBOT = "CBOT"  # Treasuries, Grains
    ICE = "ICE"  # Brent, Coffee


class OrderSide(str, Enum):
    """Order side for futures trading."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type for futures trading."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TRAILING_STOP_MARKET = "TRAILING_STOP_MARKET"


class TimeInForce(str, Enum):
    """Time in force for futures orders."""

    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    GTX = "GTX"  # Good Till Crossing (Post-Only)
    DAY = "DAY"  # Day order (CME)


class WorkingType(str, Enum):
    """Price type for stop orders (Binance)."""

    MARK_PRICE = "MARK_PRICE"
    CONTRACT_PRICE = "CONTRACT_PRICE"


class OrderStatus(str, Enum):
    """Order status for futures orders."""

    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    PENDING_NEW = "PENDING_NEW"
    PENDING_CANCEL = "PENDING_CANCEL"


# ============================================================================
# CONTRACT SPECIFICATION (Works for ALL futures types)
# ============================================================================


@dataclass(frozen=True)
class FuturesContractSpec:
    """
    Unified futures contract specification.

    Works for Crypto (Binance) AND CME futures.
    This dataclass contains all static contract properties.

    Attributes:
        symbol: Trading symbol (BTCUSDT, ES, GC, 6E)
        futures_type: Classification (CRYPTO_PERPETUAL, INDEX_FUTURES, etc.)
        contract_type: Expiration type (PERPETUAL, CURRENT_QUARTER, etc.)
        exchange: Exchange where contract trades
        base_asset: Base asset symbol (BTC, SPX, Gold, EUR)
        quote_asset: Quote asset symbol (USDT, USD)
        margin_asset: Asset used for margin (USDT, USD)
        contract_size: Units per contract (typically 1 for crypto)
        multiplier: Price multiplier (ES=$50, GC=100oz)
        tick_size: Minimum price increment
        tick_value: Value per tick move (ES=$12.50)
        min_qty: Minimum order quantity
        max_qty: Maximum order quantity
        lot_size: Order quantity increment
        max_leverage: Maximum allowed leverage
        initial_margin_pct: Initial margin as % of notional
        maint_margin_pct: Maintenance margin as % of notional
        settlement_type: Settlement method
        delivery_date: Delivery date for expiring contracts
        last_trading_day: Last day to trade
        liquidation_fee_pct: Fee charged on liquidation
        trading_hours: Trading hours description

    Examples:
        >>> btc_perp = FuturesContractSpec(
        ...     symbol="BTCUSDT",
        ...     futures_type=FuturesType.CRYPTO_PERPETUAL,
        ...     contract_type=ContractType.PERPETUAL,
        ...     exchange=Exchange.BINANCE,
        ...     base_asset="BTC",
        ...     quote_asset="USDT",
        ...     margin_asset="USDT",
        ...     max_leverage=125,
        ... )
        >>> btc_perp.is_perpetual
        True

        >>> es_fut = FuturesContractSpec(
        ...     symbol="ES",
        ...     futures_type=FuturesType.INDEX_FUTURES,
        ...     contract_type=ContractType.CURRENT_QUARTER,
        ...     exchange=Exchange.CME,
        ...     base_asset="SPX",
        ...     quote_asset="USD",
        ...     margin_asset="USD",
        ...     multiplier=Decimal("50"),
        ...     tick_size=Decimal("0.25"),
        ...     tick_value=Decimal("12.50"),
        ...     max_leverage=20,
        ... )
        >>> es_fut.notional_at_price(Decimal("5000"))
        Decimal('250000')
    """

    # Core identification
    symbol: str  # BTCUSDT, ES, GC, 6E
    futures_type: FuturesType  # Classification
    contract_type: ContractType  # Perpetual, quarterly, monthly
    exchange: Exchange  # Where it trades

    # Asset identifiers
    base_asset: str  # BTC, SPX, Gold, EUR
    quote_asset: str  # USDT, USD
    margin_asset: str  # USDT, USD

    # Contract sizing
    contract_size: Decimal = Decimal("1")  # Units per contract
    multiplier: Decimal = Decimal("1")  # Price multiplier (ES=$50, GC=100oz)
    tick_size: Decimal = Decimal("0.01")  # Minimum price increment
    tick_value: Decimal = Decimal("0.01")  # Value per tick (ES=$12.50)

    # Position limits
    min_qty: Decimal = Decimal("0.001")
    max_qty: Decimal = Decimal("1000000")
    lot_size: Decimal = Decimal("0.001")  # Order quantity increment

    # Margin requirements
    max_leverage: int = 125  # Crypto: 125x, ES: ~20x, GC: ~10x
    initial_margin_pct: Decimal = Decimal("5.0")  # % of notional
    maint_margin_pct: Decimal = Decimal("4.0")  # % of notional

    # Settlement
    settlement_type: SettlementType = SettlementType.CASH
    delivery_date: Optional[str] = None  # For expiring contracts (YYYYMMDD)
    last_trading_day: Optional[str] = None  # Last day to trade

    # Fees
    liquidation_fee_pct: Decimal = Decimal("0.5")  # Crypto: 0.5%, CME: varies
    maker_fee_bps: Decimal = Decimal("2.0")  # Maker fee in basis points
    taker_fee_bps: Decimal = Decimal("4.0")  # Taker fee in basis points

    # Trading hours
    trading_hours: str = "24/7"  # "24/7", "23/5", "RTH only"

    @property
    def notional_per_contract(self) -> Decimal:
        """
        Calculate base notional value per contract (price-independent).

        For price-dependent notional, use notional_at_price().
        """
        return self.multiplier * self.contract_size

    @property
    def is_perpetual(self) -> bool:
        """Check if perpetual (no expiry)."""
        return self.contract_type == ContractType.PERPETUAL

    @property
    def is_crypto(self) -> bool:
        """Check if crypto futures."""
        return self.futures_type.is_crypto

    @property
    def uses_funding(self) -> bool:
        """Check if uses funding rate instead of settlement."""
        return self.settlement_type == SettlementType.FUNDING

    def notional_at_price(self, price: Decimal) -> Decimal:
        """
        Calculate notional value at a given price.

        Args:
            price: Current price

        Returns:
            Notional value = price * multiplier * contract_size
        """
        return price * self.multiplier * self.contract_size

    def tick_to_price(self, ticks: int) -> Decimal:
        """
        Convert tick count to price change.

        Args:
            ticks: Number of ticks

        Returns:
            Price change = ticks * tick_size
        """
        return Decimal(ticks) * self.tick_size

    def price_to_ticks(self, price_change: Decimal) -> int:
        """
        Convert price change to tick count.

        Args:
            price_change: Price difference

        Returns:
            Number of ticks (floored)
        """
        if self.tick_size == 0:
            return 0
        return int(price_change / self.tick_size)

    def round_price(self, price: Decimal) -> Decimal:
        """
        Round price to valid tick size.

        Args:
            price: Raw price

        Returns:
            Price rounded to nearest tick
        """
        if self.tick_size <= 0:
            return price
        return (price / self.tick_size).quantize(Decimal("1"), rounding=ROUND_DOWN) * self.tick_size

    def round_qty(self, qty: Decimal) -> Decimal:
        """
        Round quantity to valid lot size.

        Args:
            qty: Raw quantity

        Returns:
            Quantity rounded to nearest lot
        """
        if self.lot_size <= 0:
            return qty
        return (qty / self.lot_size).quantize(Decimal("1"), rounding=ROUND_DOWN) * self.lot_size

    def validate_order(
        self, qty: Decimal, price: Optional[Decimal] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate order parameters against contract spec.

        Args:
            qty: Order quantity
            price: Order price (optional for market orders)

        Returns:
            (is_valid, error_message)
        """
        abs_qty = abs(qty)

        if abs_qty < self.min_qty:
            return False, f"Quantity {abs_qty} below minimum {self.min_qty}"

        if abs_qty > self.max_qty:
            return False, f"Quantity {abs_qty} exceeds maximum {self.max_qty}"

        # Check lot size
        remainder = abs_qty % self.lot_size
        if remainder != 0:
            return False, f"Quantity {abs_qty} not divisible by lot size {self.lot_size}"

        # Check tick size for limit orders
        if price is not None and self.tick_size > 0:
            remainder = price % self.tick_size
            if remainder != 0:
                return False, f"Price {price} not divisible by tick size {self.tick_size}"

        return True, None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "futures_type": self.futures_type.value,
            "contract_type": self.contract_type.value,
            "exchange": self.exchange.value,
            "base_asset": self.base_asset,
            "quote_asset": self.quote_asset,
            "margin_asset": self.margin_asset,
            "contract_size": str(self.contract_size),
            "multiplier": str(self.multiplier),
            "tick_size": str(self.tick_size),
            "tick_value": str(self.tick_value),
            "min_qty": str(self.min_qty),
            "max_qty": str(self.max_qty),
            "lot_size": str(self.lot_size),
            "max_leverage": self.max_leverage,
            "initial_margin_pct": str(self.initial_margin_pct),
            "maint_margin_pct": str(self.maint_margin_pct),
            "settlement_type": self.settlement_type.value,
            "delivery_date": self.delivery_date,
            "last_trading_day": self.last_trading_day,
            "liquidation_fee_pct": str(self.liquidation_fee_pct),
            "maker_fee_bps": str(self.maker_fee_bps),
            "taker_fee_bps": str(self.taker_fee_bps),
            "trading_hours": self.trading_hours,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FuturesContractSpec":
        """Create from dictionary."""

        def _to_decimal(v: Any, default: str = "0") -> Decimal:
            if v is None:
                return Decimal(default)
            try:
                return Decimal(str(v))
            except Exception:
                return Decimal(default)

        return cls(
            symbol=str(d.get("symbol", "")),
            futures_type=FuturesType(d.get("futures_type", "CRYPTO_PERPETUAL")),
            contract_type=ContractType(d.get("contract_type", "PERPETUAL")),
            exchange=Exchange(d.get("exchange", "BINANCE")),
            base_asset=str(d.get("base_asset", "")),
            quote_asset=str(d.get("quote_asset", "")),
            margin_asset=str(d.get("margin_asset", "")),
            contract_size=_to_decimal(d.get("contract_size"), "1"),
            multiplier=_to_decimal(d.get("multiplier"), "1"),
            tick_size=_to_decimal(d.get("tick_size"), "0.01"),
            tick_value=_to_decimal(d.get("tick_value"), "0.01"),
            min_qty=_to_decimal(d.get("min_qty"), "0.001"),
            max_qty=_to_decimal(d.get("max_qty"), "1000000"),
            lot_size=_to_decimal(d.get("lot_size"), "0.001"),
            max_leverage=int(d.get("max_leverage", 125)),
            initial_margin_pct=_to_decimal(d.get("initial_margin_pct"), "5.0"),
            maint_margin_pct=_to_decimal(d.get("maint_margin_pct"), "4.0"),
            settlement_type=SettlementType(d.get("settlement_type", "CASH")),
            delivery_date=d.get("delivery_date"),
            last_trading_day=d.get("last_trading_day"),
            liquidation_fee_pct=_to_decimal(d.get("liquidation_fee_pct"), "0.5"),
            maker_fee_bps=_to_decimal(d.get("maker_fee_bps"), "2.0"),
            taker_fee_bps=_to_decimal(d.get("taker_fee_bps"), "4.0"),
            trading_hours=str(d.get("trading_hours", "24/7")),
        )


# ============================================================================
# LEVERAGE BRACKET (Binance tiered margin)
# ============================================================================


@dataclass(frozen=True)
class LeverageBracket:
    """
    Leverage bracket for tiered margin calculation (Binance).

    Binance uses tiered brackets where larger positions require more margin.

    Attributes:
        bracket: Bracket number (1, 2, 3, ...)
        notional_cap: Maximum notional for this bracket (USDT)
        maint_margin_rate: Maintenance margin rate (e.g., 0.004 = 0.4%)
        max_leverage: Maximum leverage in this bracket
        cum_maintenance: Cumulative maintenance (for margin calculation)

    Example:
        Bracket 1: notional_cap=10000, maint_margin_rate=0.004, max_leverage=125
        Bracket 2: notional_cap=50000, maint_margin_rate=0.005, max_leverage=100
        ...
    """

    bracket: int
    notional_cap: Decimal  # Maximum position notional
    maint_margin_rate: Decimal  # Maintenance margin rate (e.g., 0.004)
    max_leverage: int  # Maximum leverage allowed
    cum_maintenance: Decimal = Decimal("0")  # Cumulative for margin calc

    @property
    def maint_margin_pct(self) -> Decimal:
        """Maintenance margin as percentage."""
        return self.maint_margin_rate * 100

    @property
    def initial_margin_rate(self) -> Decimal:
        """Initial margin rate = 1 / max_leverage."""
        if self.max_leverage <= 0:
            return Decimal("1")
        return Decimal("1") / Decimal(self.max_leverage)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bracket": self.bracket,
            "notional_cap": str(self.notional_cap),
            "maint_margin_rate": str(self.maint_margin_rate),
            "max_leverage": self.max_leverage,
            "cum_maintenance": str(self.cum_maintenance),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LeverageBracket":
        """Create from dictionary."""
        return cls(
            bracket=int(d.get("bracket", 1)),
            notional_cap=Decimal(str(d.get("notional_cap", "0"))),
            maint_margin_rate=Decimal(str(d.get("maint_margin_rate", "0.004"))),
            max_leverage=int(d.get("max_leverage", 125)),
            cum_maintenance=Decimal(str(d.get("cum_maintenance", "0"))),
        )


# ============================================================================
# FUTURES POSITION
# ============================================================================


@dataclass(frozen=True)
class FuturesPosition:
    """
    Current futures position state.

    Represents the current state of a position including entry price,
    P&L, margin, and liquidation price.

    Attributes:
        symbol: Contract symbol
        side: Position side (BOTH for one-way, LONG/SHORT for hedge)
        entry_price: Average entry price
        qty: Position quantity (positive for long, negative for short)
        leverage: Current leverage setting
        margin_mode: Cross or isolated margin
        unrealized_pnl: Unrealized profit/loss
        realized_pnl: Realized profit/loss
        liquidation_price: Estimated liquidation price
        mark_price: Current mark price
        margin: Isolated margin amount (for isolated mode)
        maint_margin: Required maintenance margin
        timestamp_ms: Last update timestamp
        position_value: Position notional value at mark price
    """

    symbol: str
    side: PositionSide
    entry_price: Decimal
    qty: Decimal  # Positive for long, negative for short
    leverage: int
    margin_mode: MarginMode
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    liquidation_price: Decimal = Decimal("0")
    mark_price: Decimal = Decimal("0")
    margin: Decimal = Decimal("0")  # Isolated margin amount
    maint_margin: Decimal = Decimal("0")
    timestamp_ms: int = 0
    position_value: Decimal = Decimal("0")  # notional at mark price

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.qty > 0 or self.side == PositionSide.LONG

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.qty < 0 or self.side == PositionSide.SHORT

    @property
    def is_open(self) -> bool:
        """Check if position is open (non-zero quantity)."""
        return self.qty != 0

    @property
    def abs_qty(self) -> Decimal:
        """Absolute quantity."""
        return abs(self.qty)

    @property
    def notional(self) -> Decimal:
        """Position notional value at entry price."""
        return self.abs_qty * self.entry_price

    @property
    def roe_pct(self) -> Decimal:
        """Return on equity percentage."""
        if self.margin == 0:
            return Decimal("0")
        return (self.unrealized_pnl / self.margin) * 100

    def calculate_pnl(self, current_price: Decimal) -> Decimal:
        """
        Calculate unrealized P&L at given price.

        Args:
            current_price: Current market/mark price

        Returns:
            Unrealized P&L
        """
        if self.qty == 0:
            return Decimal("0")

        price_diff = current_price - self.entry_price
        return price_diff * self.qty

    def calculate_roe(self, current_price: Decimal) -> Decimal:
        """
        Calculate ROE at given price.

        Args:
            current_price: Current market/mark price

        Returns:
            ROE as decimal (not percentage)
        """
        if self.margin == 0:
            return Decimal("0")
        pnl = self.calculate_pnl(current_price)
        return pnl / self.margin

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_price": str(self.entry_price),
            "qty": str(self.qty),
            "leverage": self.leverage,
            "margin_mode": self.margin_mode.value,
            "unrealized_pnl": str(self.unrealized_pnl),
            "realized_pnl": str(self.realized_pnl),
            "liquidation_price": str(self.liquidation_price),
            "mark_price": str(self.mark_price),
            "margin": str(self.margin),
            "maint_margin": str(self.maint_margin),
            "timestamp_ms": self.timestamp_ms,
            "position_value": str(self.position_value),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FuturesPosition":
        """Create from dictionary."""
        return cls(
            symbol=str(d.get("symbol", "")),
            side=PositionSide(d.get("side", "BOTH")),
            entry_price=Decimal(str(d.get("entry_price", "0"))),
            qty=Decimal(str(d.get("qty", "0"))),
            leverage=int(d.get("leverage", 1)),
            margin_mode=MarginMode(d.get("margin_mode", "CROSS")),
            unrealized_pnl=Decimal(str(d.get("unrealized_pnl", "0"))),
            realized_pnl=Decimal(str(d.get("realized_pnl", "0"))),
            liquidation_price=Decimal(str(d.get("liquidation_price", "0"))),
            mark_price=Decimal(str(d.get("mark_price", "0"))),
            margin=Decimal(str(d.get("margin", "0"))),
            maint_margin=Decimal(str(d.get("maint_margin", "0"))),
            timestamp_ms=int(d.get("timestamp_ms", 0)),
            position_value=Decimal(str(d.get("position_value", "0"))),
        )


# ============================================================================
# MARGIN REQUIREMENT
# ============================================================================


@dataclass(frozen=True)
class MarginRequirement:
    """
    Margin requirements for a position or order.

    Attributes:
        initial: Initial margin required to open position
        maintenance: Maintenance margin to keep position open
        variation: Daily variation margin (CME)
        available: Available margin after requirements
    """

    initial: Decimal
    maintenance: Decimal
    variation: Decimal = Decimal("0")
    available: Decimal = Decimal("0")

    @property
    def margin_ratio(self) -> Decimal:
        """Maintenance to initial ratio."""
        if self.initial == 0:
            return Decimal("0")
        return self.maintenance / self.initial

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "initial": str(self.initial),
            "maintenance": str(self.maintenance),
            "variation": str(self.variation),
            "available": str(self.available),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MarginRequirement":
        """Create from dictionary."""
        return cls(
            initial=Decimal(str(d.get("initial", "0"))),
            maintenance=Decimal(str(d.get("maintenance", "0"))),
            variation=Decimal(str(d.get("variation", "0"))),
            available=Decimal(str(d.get("available", "0"))),
        )


# ============================================================================
# FUNDING PAYMENT (Crypto Perpetual)
# ============================================================================


@dataclass(frozen=True)
class FundingPayment:
    """
    Funding rate payment record (crypto perpetual only).

    Funding payments occur every 8 hours on Binance and transfer
    between longs and shorts to keep perpetual price near index.

    Attributes:
        symbol: Contract symbol
        timestamp_ms: Payment timestamp
        funding_rate: Rate for this interval (e.g., 0.0001 = 0.01%)
        mark_price: Mark price at funding time
        position_qty: Position quantity at funding time
        payment_amount: Payment received (positive) or paid (negative)
        asset: Settlement asset (USDT)
    """

    symbol: str
    timestamp_ms: int
    funding_rate: Decimal  # e.g., 0.0001 = 0.01%
    mark_price: Decimal
    position_qty: Decimal
    payment_amount: Decimal  # Positive = received, negative = paid
    asset: str = "USDT"

    @property
    def rate_bps(self) -> Decimal:
        """Funding rate in basis points."""
        return self.funding_rate * 10000

    @property
    def rate_pct(self) -> Decimal:
        """Funding rate as percentage."""
        return self.funding_rate * 100

    @property
    def annualized_rate(self) -> Decimal:
        """Annualized funding rate (3 payments per day * 365)."""
        return self.funding_rate * 3 * 365 * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp_ms": self.timestamp_ms,
            "funding_rate": str(self.funding_rate),
            "mark_price": str(self.mark_price),
            "position_qty": str(self.position_qty),
            "payment_amount": str(self.payment_amount),
            "asset": self.asset,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FundingPayment":
        """Create from dictionary."""
        return cls(
            symbol=str(d.get("symbol", "")),
            timestamp_ms=int(d.get("timestamp_ms", 0)),
            funding_rate=Decimal(str(d.get("funding_rate", "0"))),
            mark_price=Decimal(str(d.get("mark_price", "0"))),
            position_qty=Decimal(str(d.get("position_qty", "0"))),
            payment_amount=Decimal(str(d.get("payment_amount", "0"))),
            asset=str(d.get("asset", "USDT")),
        )


# ============================================================================
# FUNDING RATE INFO
# ============================================================================


@dataclass(frozen=True)
class FundingRateInfo:
    """
    Current funding rate information.

    Attributes:
        symbol: Contract symbol
        funding_rate: Current funding rate
        next_funding_time_ms: Next funding timestamp
        mark_price: Current mark price
        index_price: Current index price
        estimated_rate: Estimated next funding rate
        funding_interval_hours: Hours between funding (8 for Binance)
    """

    symbol: str
    funding_rate: Decimal
    next_funding_time_ms: int
    mark_price: Decimal = Decimal("0")
    index_price: Decimal = Decimal("0")
    estimated_rate: Optional[Decimal] = None
    funding_interval_hours: int = 8

    @property
    def premium_index(self) -> Decimal:
        """Premium index = (mark_price - index_price) / index_price."""
        if self.index_price == 0:
            return Decimal("0")
        return (self.mark_price - self.index_price) / self.index_price

    @property
    def time_to_funding_ms(self) -> int:
        """Milliseconds until next funding (based on current time)."""
        import time

        now_ms = int(time.time() * 1000)
        return max(0, self.next_funding_time_ms - now_ms)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "funding_rate": str(self.funding_rate),
            "next_funding_time_ms": self.next_funding_time_ms,
            "mark_price": str(self.mark_price),
            "index_price": str(self.index_price),
            "estimated_rate": str(self.estimated_rate) if self.estimated_rate else None,
            "funding_interval_hours": self.funding_interval_hours,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FundingRateInfo":
        """Create from dictionary."""
        est_rate = d.get("estimated_rate")
        return cls(
            symbol=str(d.get("symbol", "")),
            funding_rate=Decimal(str(d.get("funding_rate", "0"))),
            next_funding_time_ms=int(d.get("next_funding_time_ms", 0)),
            mark_price=Decimal(str(d.get("mark_price", "0"))),
            index_price=Decimal(str(d.get("index_price", "0"))),
            estimated_rate=Decimal(str(est_rate)) if est_rate else None,
            funding_interval_hours=int(d.get("funding_interval_hours", 8)),
        )


# ============================================================================
# LIQUIDATION EVENT
# ============================================================================


@dataclass(frozen=True)
class LiquidationEvent:
    """
    Liquidation event record.

    Records when a position is liquidated due to insufficient margin.

    Attributes:
        symbol: Contract symbol
        timestamp_ms: Liquidation timestamp
        side: Closing side (BUY to close short, SELL to close long)
        qty: Liquidated quantity
        price: Liquidation price
        liquidation_type: "partial" or "full"
        loss_amount: P&L loss from liquidation
        insurance_fund_contribution: Amount to/from insurance fund
        order_id: Liquidation order ID
    """

    symbol: str
    timestamp_ms: int
    side: str  # "BUY" or "SELL" (closing side)
    qty: Decimal
    price: Decimal
    liquidation_type: str  # "partial" or "full"
    loss_amount: Decimal
    insurance_fund_contribution: Decimal = Decimal("0")
    order_id: Optional[str] = None

    @property
    def is_full_liquidation(self) -> bool:
        """Check if full position was liquidated."""
        return self.liquidation_type.lower() == "full"

    @property
    def notional(self) -> Decimal:
        """Liquidation notional value."""
        return self.qty * self.price

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp_ms": self.timestamp_ms,
            "side": self.side,
            "qty": str(self.qty),
            "price": str(self.price),
            "liquidation_type": self.liquidation_type,
            "loss_amount": str(self.loss_amount),
            "insurance_fund_contribution": str(self.insurance_fund_contribution),
            "order_id": self.order_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LiquidationEvent":
        """Create from dictionary."""
        return cls(
            symbol=str(d.get("symbol", "")),
            timestamp_ms=int(d.get("timestamp_ms", 0)),
            side=str(d.get("side", "")),
            qty=Decimal(str(d.get("qty", "0"))),
            price=Decimal(str(d.get("price", "0"))),
            liquidation_type=str(d.get("liquidation_type", "full")),
            loss_amount=Decimal(str(d.get("loss_amount", "0"))),
            insurance_fund_contribution=Decimal(str(d.get("insurance_fund_contribution", "0"))),
            order_id=d.get("order_id"),
        )


# ============================================================================
# OPEN INTEREST
# ============================================================================


@dataclass(frozen=True)
class OpenInterestInfo:
    """
    Open interest information.

    Attributes:
        symbol: Contract symbol
        open_interest: Total open interest in contracts
        open_interest_value: Value in quote currency
        timestamp_ms: Data timestamp
    """

    symbol: str
    open_interest: Decimal
    open_interest_value: Decimal = Decimal("0")
    timestamp_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "open_interest": str(self.open_interest),
            "open_interest_value": str(self.open_interest_value),
            "timestamp_ms": self.timestamp_ms,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OpenInterestInfo":
        """Create from dictionary."""
        return cls(
            symbol=str(d.get("symbol", "")),
            open_interest=Decimal(str(d.get("open_interest", "0"))),
            open_interest_value=Decimal(str(d.get("open_interest_value", "0"))),
            timestamp_ms=int(d.get("timestamp_ms", 0)),
        )


# ============================================================================
# FUTURES ACCOUNT STATE
# ============================================================================


@dataclass(frozen=True)
class FuturesAccountState:
    """
    Futures account state snapshot.

    Comprehensive account state including balances, margins, and positions.

    Attributes:
        timestamp_ms: Snapshot timestamp
        total_wallet_balance: Total wallet balance
        total_margin_balance: Margin balance (wallet + unrealized P&L)
        total_unrealized_pnl: Sum of unrealized P&L across positions
        available_balance: Available for new positions
        total_initial_margin: Total initial margin used
        total_maint_margin: Total maintenance margin required
        total_position_initial_margin: Position initial margin
        total_open_order_initial_margin: Open order margin
        max_withdraw_amount: Maximum withdrawable amount
        positions: Dict of symbol to FuturesPosition
        asset: Account asset (USDT, USD)
    """

    timestamp_ms: int
    total_wallet_balance: Decimal
    total_margin_balance: Decimal
    total_unrealized_pnl: Decimal
    available_balance: Decimal
    total_initial_margin: Decimal
    total_maint_margin: Decimal
    total_position_initial_margin: Decimal = Decimal("0")
    total_open_order_initial_margin: Decimal = Decimal("0")
    max_withdraw_amount: Decimal = Decimal("0")
    positions: Dict[str, FuturesPosition] = field(default_factory=dict)
    asset: str = "USDT"

    @property
    def margin_ratio(self) -> Decimal:
        """
        Margin ratio = maint_margin / margin_balance.

        Higher ratio = closer to liquidation.
        """
        if self.total_margin_balance == 0:
            return Decimal("0")
        return self.total_maint_margin / self.total_margin_balance

    @property
    def margin_level_pct(self) -> Decimal:
        """
        Margin level as percentage.

        100% = at maintenance, <100% = liquidation.
        """
        if self.total_maint_margin == 0:
            return Decimal("100")
        return (self.total_margin_balance / self.total_maint_margin) * 100

    @property
    def is_at_risk(self) -> bool:
        """Check if account is at margin call risk (>80% margin ratio)."""
        return self.margin_ratio > Decimal("0.8")

    @property
    def num_positions(self) -> int:
        """Number of open positions."""
        return sum(1 for p in self.positions.values() if p.is_open)

    @property
    def total_position_notional(self) -> Decimal:
        """Total notional value of all positions."""
        return sum(p.position_value for p in self.positions.values())

    def get_position(self, symbol: str) -> Optional[FuturesPosition]:
        """Get position for symbol."""
        return self.positions.get(symbol)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp_ms": self.timestamp_ms,
            "total_wallet_balance": str(self.total_wallet_balance),
            "total_margin_balance": str(self.total_margin_balance),
            "total_unrealized_pnl": str(self.total_unrealized_pnl),
            "available_balance": str(self.available_balance),
            "total_initial_margin": str(self.total_initial_margin),
            "total_maint_margin": str(self.total_maint_margin),
            "total_position_initial_margin": str(self.total_position_initial_margin),
            "total_open_order_initial_margin": str(self.total_open_order_initial_margin),
            "max_withdraw_amount": str(self.max_withdraw_amount),
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "asset": self.asset,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FuturesAccountState":
        """Create from dictionary."""
        positions_data = d.get("positions", {})
        positions = {k: FuturesPosition.from_dict(v) for k, v in positions_data.items()}

        return cls(
            timestamp_ms=int(d.get("timestamp_ms", 0)),
            total_wallet_balance=Decimal(str(d.get("total_wallet_balance", "0"))),
            total_margin_balance=Decimal(str(d.get("total_margin_balance", "0"))),
            total_unrealized_pnl=Decimal(str(d.get("total_unrealized_pnl", "0"))),
            available_balance=Decimal(str(d.get("available_balance", "0"))),
            total_initial_margin=Decimal(str(d.get("total_initial_margin", "0"))),
            total_maint_margin=Decimal(str(d.get("total_maint_margin", "0"))),
            total_position_initial_margin=Decimal(str(d.get("total_position_initial_margin", "0"))),
            total_open_order_initial_margin=Decimal(
                str(d.get("total_open_order_initial_margin", "0"))
            ),
            max_withdraw_amount=Decimal(str(d.get("max_withdraw_amount", "0"))),
            positions=positions,
            asset=str(d.get("asset", "USDT")),
        )


# ============================================================================
# FUTURES ORDER
# ============================================================================


@dataclass(frozen=True)
class FuturesOrder:
    """
    Unified order representation for all futures types.

    Supports both crypto (Binance) and CME order types.

    Attributes:
        symbol: Contract symbol
        side: Order side (BUY/SELL)
        order_type: Order type (MARKET, LIMIT, etc.)
        qty: Order quantity
        price: Limit price (None for market orders)
        stop_price: Stop trigger price
        reduce_only: Only reduce position
        time_in_force: TIF (GTC, IOC, FOK, DAY)
        post_only: Maker only
        client_order_id: Client-assigned order ID
        working_type: Stop trigger price type (MARK_PRICE, CONTRACT_PRICE)
        position_side: Position side for hedge mode
        close_position: Close entire position
    """

    symbol: str
    side: OrderSide
    order_type: OrderType
    qty: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    reduce_only: bool = False
    time_in_force: TimeInForce = TimeInForce.GTC
    post_only: bool = False
    client_order_id: Optional[str] = None
    working_type: WorkingType = WorkingType.CONTRACT_PRICE
    position_side: PositionSide = PositionSide.BOTH
    close_position: bool = False

    @property
    def is_market_order(self) -> bool:
        """Check if market order."""
        return self.order_type in (
            OrderType.MARKET,
            OrderType.STOP_MARKET,
            OrderType.TAKE_PROFIT_MARKET,
        )

    @property
    def is_limit_order(self) -> bool:
        """Check if limit order."""
        return self.order_type in (OrderType.LIMIT, OrderType.STOP, OrderType.TAKE_PROFIT)

    @property
    def is_stop_order(self) -> bool:
        """Check if stop order (has trigger)."""
        return self.order_type in (
            OrderType.STOP,
            OrderType.STOP_MARKET,
            OrderType.TAKE_PROFIT,
            OrderType.TAKE_PROFIT_MARKET,
            OrderType.TRAILING_STOP_MARKET,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "qty": str(self.qty),
            "price": str(self.price) if self.price else None,
            "stop_price": str(self.stop_price) if self.stop_price else None,
            "reduce_only": self.reduce_only,
            "time_in_force": self.time_in_force.value,
            "post_only": self.post_only,
            "client_order_id": self.client_order_id,
            "working_type": self.working_type.value,
            "position_side": self.position_side.value,
            "close_position": self.close_position,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FuturesOrder":
        """Create from dictionary."""
        price = d.get("price")
        stop_price = d.get("stop_price")

        return cls(
            symbol=str(d.get("symbol", "")),
            side=OrderSide(d.get("side", "BUY")),
            order_type=OrderType(d.get("order_type", "MARKET")),
            qty=Decimal(str(d.get("qty", "0"))),
            price=Decimal(str(price)) if price else None,
            stop_price=Decimal(str(stop_price)) if stop_price else None,
            reduce_only=bool(d.get("reduce_only", False)),
            time_in_force=TimeInForce(d.get("time_in_force", "GTC")),
            post_only=bool(d.get("post_only", False)),
            client_order_id=d.get("client_order_id"),
            working_type=WorkingType(d.get("working_type", "CONTRACT_PRICE")),
            position_side=PositionSide(d.get("position_side", "BOTH")),
            close_position=bool(d.get("close_position", False)),
        )


# ============================================================================
# FUTURES FILL
# ============================================================================


@dataclass(frozen=True)
class FuturesFill:
    """
    Result of a futures order execution.

    Attributes:
        order_id: Exchange order ID
        client_order_id: Client order ID
        symbol: Contract symbol
        side: Order side
        filled_qty: Filled quantity
        avg_price: Average fill price
        commission: Trading commission
        commission_asset: Commission currency
        realized_pnl: Realized P&L from this fill
        timestamp_ms: Fill timestamp
        is_maker: True if maker fill
        liquidity: "MAKER" or "TAKER"
        margin_impact: Change in margin requirement
        new_position_size: Position after fill
        new_avg_entry: New average entry price
    """

    order_id: str
    client_order_id: Optional[str]
    symbol: str
    side: OrderSide
    filled_qty: Decimal
    avg_price: Decimal
    commission: Decimal
    commission_asset: str
    realized_pnl: Decimal
    timestamp_ms: int
    is_maker: bool
    liquidity: str  # "MAKER" or "TAKER"
    margin_impact: Decimal = Decimal("0")
    new_position_size: Decimal = Decimal("0")
    new_avg_entry: Decimal = Decimal("0")

    @property
    def notional(self) -> Decimal:
        """Fill notional value."""
        return self.filled_qty * self.avg_price

    @property
    def commission_bps(self) -> Decimal:
        """Commission in basis points."""
        if self.notional == 0:
            return Decimal("0")
        return (self.commission / self.notional) * 10000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "filled_qty": str(self.filled_qty),
            "avg_price": str(self.avg_price),
            "commission": str(self.commission),
            "commission_asset": self.commission_asset,
            "realized_pnl": str(self.realized_pnl),
            "timestamp_ms": self.timestamp_ms,
            "is_maker": self.is_maker,
            "liquidity": self.liquidity,
            "margin_impact": str(self.margin_impact),
            "new_position_size": str(self.new_position_size),
            "new_avg_entry": str(self.new_avg_entry),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FuturesFill":
        """Create from dictionary."""
        return cls(
            order_id=str(d.get("order_id", "")),
            client_order_id=d.get("client_order_id"),
            symbol=str(d.get("symbol", "")),
            side=OrderSide(d.get("side", "BUY")),
            filled_qty=Decimal(str(d.get("filled_qty", "0"))),
            avg_price=Decimal(str(d.get("avg_price", "0"))),
            commission=Decimal(str(d.get("commission", "0"))),
            commission_asset=str(d.get("commission_asset", "USDT")),
            realized_pnl=Decimal(str(d.get("realized_pnl", "0"))),
            timestamp_ms=int(d.get("timestamp_ms", 0)),
            is_maker=bool(d.get("is_maker", False)),
            liquidity=str(d.get("liquidity", "TAKER")),
            margin_impact=Decimal(str(d.get("margin_impact", "0"))),
            new_position_size=Decimal(str(d.get("new_position_size", "0"))),
            new_avg_entry=Decimal(str(d.get("new_avg_entry", "0"))),
        )


# ============================================================================
# MARK PRICE TICK
# ============================================================================


@dataclass(frozen=True)
class MarkPriceTick:
    """
    Real-time mark price update.

    Attributes:
        symbol: Contract symbol
        mark_price: Current mark price
        index_price: Current index price
        estimated_settle_price: Estimated next settlement price
        funding_rate: Current funding rate
        next_funding_time_ms: Next funding timestamp
        timestamp_ms: Update timestamp
    """

    symbol: str
    mark_price: Decimal
    index_price: Decimal
    estimated_settle_price: Decimal = Decimal("0")
    funding_rate: Decimal = Decimal("0")
    next_funding_time_ms: int = 0
    timestamp_ms: int = 0

    @property
    def basis_bps(self) -> Decimal:
        """Basis in basis points = (mark - index) / index * 10000."""
        if self.index_price == 0:
            return Decimal("0")
        return (self.mark_price - self.index_price) / self.index_price * 10000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "mark_price": str(self.mark_price),
            "index_price": str(self.index_price),
            "estimated_settle_price": str(self.estimated_settle_price),
            "funding_rate": str(self.funding_rate),
            "next_funding_time_ms": self.next_funding_time_ms,
            "timestamp_ms": self.timestamp_ms,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MarkPriceTick":
        """Create from dictionary."""
        return cls(
            symbol=str(d.get("symbol", "")),
            mark_price=Decimal(str(d.get("mark_price", "0"))),
            index_price=Decimal(str(d.get("index_price", "0"))),
            estimated_settle_price=Decimal(str(d.get("estimated_settle_price", "0"))),
            funding_rate=Decimal(str(d.get("funding_rate", "0"))),
            next_funding_time_ms=int(d.get("next_funding_time_ms", 0)),
            timestamp_ms=int(d.get("timestamp_ms", 0)),
        )


# ============================================================================
# SETTLEMENT INFO (CME)
# ============================================================================


@dataclass(frozen=True)
class SettlementInfo:
    """
    Settlement information for expiring contracts (CME).

    Attributes:
        symbol: Contract symbol
        settlement_type: Cash or physical
        settlement_date: Settlement date (YYYYMMDD)
        settlement_price: Final settlement price
        settlement_time: Settlement time
        delivery_start: Delivery period start
        delivery_end: Delivery period end
    """

    symbol: str
    settlement_type: SettlementType
    settlement_date: str
    settlement_price: Decimal = Decimal("0")
    settlement_time: Optional[str] = None
    delivery_start: Optional[str] = None
    delivery_end: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "settlement_type": self.settlement_type.value,
            "settlement_date": self.settlement_date,
            "settlement_price": str(self.settlement_price),
            "settlement_time": self.settlement_time,
            "delivery_start": self.delivery_start,
            "delivery_end": self.delivery_end,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SettlementInfo":
        """Create from dictionary."""
        return cls(
            symbol=str(d.get("symbol", "")),
            settlement_type=SettlementType(d.get("settlement_type", "CASH")),
            settlement_date=str(d.get("settlement_date", "")),
            settlement_price=Decimal(str(d.get("settlement_price", "0"))),
            settlement_time=d.get("settlement_time"),
            delivery_start=d.get("delivery_start"),
            delivery_end=d.get("delivery_end"),
        )


# ============================================================================
# CONTRACT ROLLOVER
# ============================================================================


@dataclass(frozen=True)
class ContractRollover:
    """
    Contract rollover information.

    Used when rolling from expiring contract to next month/quarter.

    Attributes:
        from_contract: Expiring contract symbol
        to_contract: New contract symbol
        roll_date: Recommended roll date
        price_adjustment: Price difference for continuous contract
        volume_adjustment: Volume adjustment factor
    """

    from_contract: str
    to_contract: str
    roll_date: str
    price_adjustment: Decimal = Decimal("0")
    volume_adjustment: Decimal = Decimal("1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "from_contract": self.from_contract,
            "to_contract": self.to_contract,
            "roll_date": self.roll_date,
            "price_adjustment": str(self.price_adjustment),
            "volume_adjustment": str(self.volume_adjustment),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ContractRollover":
        """Create from dictionary."""
        return cls(
            from_contract=str(d.get("from_contract", "")),
            to_contract=str(d.get("to_contract", "")),
            roll_date=str(d.get("roll_date", "")),
            price_adjustment=Decimal(str(d.get("price_adjustment", "0"))),
            volume_adjustment=Decimal(str(d.get("volume_adjustment", "1"))),
        )


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def create_btc_perpetual_spec() -> FuturesContractSpec:
    """Create BTC perpetual contract specification (Binance defaults)."""
    return FuturesContractSpec(
        symbol="BTCUSDT",
        futures_type=FuturesType.CRYPTO_PERPETUAL,
        contract_type=ContractType.PERPETUAL,
        exchange=Exchange.BINANCE,
        base_asset="BTC",
        quote_asset="USDT",
        margin_asset="USDT",
        contract_size=Decimal("1"),
        multiplier=Decimal("1"),
        tick_size=Decimal("0.10"),
        tick_value=Decimal("0.10"),
        min_qty=Decimal("0.001"),
        max_qty=Decimal("1000"),
        lot_size=Decimal("0.001"),
        max_leverage=125,
        initial_margin_pct=Decimal("0.8"),  # 125x → 0.8%
        maint_margin_pct=Decimal("0.4"),
        settlement_type=SettlementType.FUNDING,
        liquidation_fee_pct=Decimal("0.5"),
        maker_fee_bps=Decimal("2.0"),
        taker_fee_bps=Decimal("4.0"),
        trading_hours="24/7",
    )


def create_eth_perpetual_spec() -> FuturesContractSpec:
    """Create ETH perpetual contract specification (Binance defaults)."""
    return FuturesContractSpec(
        symbol="ETHUSDT",
        futures_type=FuturesType.CRYPTO_PERPETUAL,
        contract_type=ContractType.PERPETUAL,
        exchange=Exchange.BINANCE,
        base_asset="ETH",
        quote_asset="USDT",
        margin_asset="USDT",
        contract_size=Decimal("1"),
        multiplier=Decimal("1"),
        tick_size=Decimal("0.01"),
        tick_value=Decimal("0.01"),
        min_qty=Decimal("0.001"),
        max_qty=Decimal("10000"),
        lot_size=Decimal("0.001"),
        max_leverage=100,
        initial_margin_pct=Decimal("1.0"),  # 100x → 1%
        maint_margin_pct=Decimal("0.5"),
        settlement_type=SettlementType.FUNDING,
        liquidation_fee_pct=Decimal("0.5"),
        maker_fee_bps=Decimal("2.0"),
        taker_fee_bps=Decimal("4.0"),
        trading_hours="24/7",
    )


def create_es_futures_spec() -> FuturesContractSpec:
    """Create E-mini S&P 500 futures contract specification (CME)."""
    return FuturesContractSpec(
        symbol="ES",
        futures_type=FuturesType.INDEX_FUTURES,
        contract_type=ContractType.CURRENT_QUARTER,
        exchange=Exchange.CME,
        base_asset="SPX",
        quote_asset="USD",
        margin_asset="USD",
        contract_size=Decimal("1"),
        multiplier=Decimal("50"),  # $50 per point
        tick_size=Decimal("0.25"),  # Minimum price increment
        tick_value=Decimal("12.50"),  # $12.50 per tick
        min_qty=Decimal("1"),
        max_qty=Decimal("10000"),
        lot_size=Decimal("1"),
        max_leverage=20,
        initial_margin_pct=Decimal("5.0"),  # ~5% initial margin
        maint_margin_pct=Decimal("4.0"),  # ~4% maintenance
        settlement_type=SettlementType.CASH,
        maker_fee_bps=Decimal("0.5"),
        taker_fee_bps=Decimal("1.0"),
        trading_hours="23/5",
    )


def create_gc_futures_spec() -> FuturesContractSpec:
    """Create Gold futures contract specification (COMEX)."""
    return FuturesContractSpec(
        symbol="GC",
        futures_type=FuturesType.COMMODITY_FUTURES,
        contract_type=ContractType.CURRENT_MONTH,
        exchange=Exchange.COMEX,
        base_asset="Gold",
        quote_asset="USD",
        margin_asset="USD",
        contract_size=Decimal("100"),  # 100 troy ounces
        multiplier=Decimal("1"),
        tick_size=Decimal("0.10"),  # $0.10 per oz
        tick_value=Decimal("10.00"),  # $10 per tick
        min_qty=Decimal("1"),
        max_qty=Decimal("6000"),
        lot_size=Decimal("1"),
        max_leverage=10,
        initial_margin_pct=Decimal("10.0"),
        maint_margin_pct=Decimal("8.0"),
        settlement_type=SettlementType.PHYSICAL,
        maker_fee_bps=Decimal("0.5"),
        taker_fee_bps=Decimal("1.0"),
        trading_hours="23/5",
    )


def create_6e_futures_spec() -> FuturesContractSpec:
    """Create Euro FX futures contract specification (CME)."""
    return FuturesContractSpec(
        symbol="6E",
        futures_type=FuturesType.CURRENCY_FUTURES,
        contract_type=ContractType.CURRENT_QUARTER,
        exchange=Exchange.CME,
        base_asset="EUR",
        quote_asset="USD",
        margin_asset="USD",
        contract_size=Decimal("125000"),  # €125,000
        multiplier=Decimal("1"),
        tick_size=Decimal("0.00005"),  # 0.5 pip
        tick_value=Decimal("6.25"),  # $6.25 per tick
        min_qty=Decimal("1"),
        max_qty=Decimal("10000"),
        lot_size=Decimal("1"),
        max_leverage=50,
        initial_margin_pct=Decimal("2.0"),
        maint_margin_pct=Decimal("1.6"),
        settlement_type=SettlementType.CASH,
        maker_fee_bps=Decimal("0.5"),
        taker_fee_bps=Decimal("1.0"),
        trading_hours="23/5",
    )


# ============================================================================
# TYPE ALIASES
# ============================================================================

# Common type aliases for futures trading
PositionDict = Dict[str, FuturesPosition]
BracketList = List[LeverageBracket]
FundingHistory = List[FundingPayment]
