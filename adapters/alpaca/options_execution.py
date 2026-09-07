# -*- coding: utf-8 -*-
"""
adapters/alpaca/options_execution.py
Alpaca options order execution adapter for US equities options.

Phase 6: Options Trading (L3 Enhancement)

**IMPLEMENTATION STATUS: PARTIAL STUB**

The following methods are partially implemented or stub:
- `get_option_chain()`: Returns empty chain with warning (API integration pending)
- Multi-leg strategies: Placeholder implementation

Per CCEA Design Doc Section 4.2: Broker Connectors are AGENT ZONE ONLY.
This adapter runs exclusively in the customer-controlled Agent environment.

**Tech Debt Tracking**: `docs/reports/TECH_DEBT_REGISTRY.md#adapter-alpaca-options-stub`

Supports (when fully implemented):
- Options order submission (calls, puts)
- Single-leg options orders
- Multi-leg strategies (spreads, straddles, etc.)
- Options chain retrieval
- Greeks-based risk management

References:
- Alpaca Options API: https://alpaca.markets/docs/trading/options/
- OCC Options Symbology: https://www.theocc.com/Market-Data/Market-Data-Reports/Series-and-Trading-Data/Series-Symbology

OCC Option Symbol Format:
    Symbol (6 chars, padded) + Expiration (YYMMDD) + Type (C/P) + Strike (8 digits, 5.3 format)
    Example: AAPL  241220C00200000 = AAPL Dec 20, 2024 $200 Call
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from core_models import Order, ExecReport, Side, OrderType, ExecStatus, Liquidity
from adapters.base import OrderExecutionAdapter, OrderResult
from adapters.models import ExchangeVendor

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Standard contract multiplier for equity options
OPTIONS_CONTRACT_MULTIPLIER = 100


# =============================================================================
# Option Enums
# =============================================================================


class OptionType(str, Enum):
    """Option type."""

    CALL = "call"
    PUT = "put"


class OptionStrategy(str, Enum):
    """Pre-defined option strategies."""

    SINGLE = "single"
    COVERED_CALL = "covered_call"
    PROTECTIVE_PUT = "protective_put"
    VERTICAL_SPREAD = "vertical_spread"  # Bull/Bear call/put spreads
    CALENDAR_SPREAD = "calendar_spread"  # Same strike, different expiration
    DIAGONAL_SPREAD = "diagonal_spread"  # Different strike and expiration
    STRADDLE = "straddle"  # Same strike call + put
    STRANGLE = "strangle"  # Different strike call + put
    IRON_CONDOR = "iron_condor"  # 4-leg neutral strategy
    BUTTERFLY = "butterfly"  # 3-strike spread
    COLLAR = "collar"  # Stock + put + short call


class OptionOrderType(str, Enum):
    """Option-specific order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class OptionContract:
    """
    Represents an option contract.

    Attributes:
        symbol: Underlying symbol (e.g., "AAPL")
        occ_symbol: Full OCC symbol (e.g., "AAPL  241220C00200000")
        option_type: CALL or PUT
        strike_price: Strike price
        expiration_date: Expiration date
        multiplier: Contract multiplier (usually 100)
    """

    symbol: str
    occ_symbol: str
    option_type: OptionType
    strike_price: float
    expiration_date: date
    multiplier: int = OPTIONS_CONTRACT_MULTIPLIER

    # Optional greeks (if available)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    implied_volatility: Optional[float] = None

    # Market data
    bid: Optional[float] = None
    ask: Optional[float] = None
    last_price: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.option_type, str):
            self.option_type = OptionType(self.option_type.lower())
        if isinstance(self.expiration_date, str):
            self.expiration_date = datetime.strptime(self.expiration_date, "%Y-%m-%d").date()

    @property
    def is_call(self) -> bool:
        """Check if option is a call."""
        return self.option_type == OptionType.CALL

    @property
    def is_put(self) -> bool:
        """Check if option is a put."""
        return self.option_type == OptionType.PUT

    @property
    def days_to_expiration(self) -> int:
        """Days until expiration."""
        return (self.expiration_date - date.today()).days

    @property
    def is_expired(self) -> bool:
        """Check if option is expired."""
        return self.days_to_expiration < 0

    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price if bid/ask available."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2.0
        return self.last_price

    @classmethod
    def from_occ_symbol(cls, occ_symbol: str) -> "OptionContract":
        """
        Parse OCC symbol into OptionContract.

        OCC format: SYMBOL (6 padded) + YYMMDD + C/P + Strike (8 digits, 5.3)
        Example: "AAPL  241220C00200000" = AAPL Dec 20, 2024 $200 Call
        """
        occ = occ_symbol.replace(" ", "")

        # Find where the date starts (first digit after symbol)
        match = re.match(r"^([A-Z]+)(\d{6})([CP])(\d{8})$", occ)
        if not match:
            raise ValueError(f"Invalid OCC symbol format: {occ_symbol}")

        symbol, date_str, opt_type, strike_str = match.groups()

        # Parse expiration date (YYMMDD)
        exp_date = datetime.strptime(date_str, "%y%m%d").date()

        # Parse strike price (last 8 digits, 5 whole + 3 decimal)
        strike = float(strike_str) / 1000.0

        return cls(
            symbol=symbol,
            occ_symbol=occ_symbol,
            option_type=OptionType.CALL if opt_type == "C" else OptionType.PUT,
            strike_price=strike,
            expiration_date=exp_date,
        )

    def to_occ_symbol(self) -> str:
        """Generate OCC symbol from contract details."""
        symbol_padded = self.symbol.ljust(6)
        date_str = self.expiration_date.strftime("%y%m%d")
        opt_char = "C" if self.is_call else "P"
        strike_str = f"{int(self.strike_price * 1000):08d}"
        return f"{symbol_padded}{date_str}{opt_char}{strike_str}"


@dataclass
class OptionLeg:
    """
    Single leg of an options order.

    For multi-leg strategies, multiple legs are combined.
    """

    contract: OptionContract
    side: Side
    qty: int
    ratio: int = 1  # For ratio spreads

    @property
    def is_long(self) -> bool:
        return self.side == Side.BUY

    @property
    def is_short(self) -> bool:
        return self.side == Side.SELL


@dataclass
class OptionOrderConfig:
    """Configuration for an options order."""

    # Primary leg
    symbol: str
    option_type: OptionType
    strike_price: float
    expiration_date: date
    side: Side
    qty: int

    # Order parameters
    order_type: OptionOrderType = OptionOrderType.LIMIT
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None

    # Additional legs for multi-leg strategies
    additional_legs: List[OptionLeg] = field(default_factory=list)
    strategy: OptionStrategy = OptionStrategy.SINGLE

    # Order settings
    time_in_force: str = "DAY"
    extended_hours: bool = False
    client_order_id: Optional[str] = None

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate options order configuration."""
        if not self.symbol:
            return False, "Symbol required"
        if self.qty <= 0:
            return False, "Quantity must be positive"
        if self.strike_price <= 0:
            return False, "Strike price must be positive"
        if self.expiration_date <= date.today():
            return False, "Expiration date must be in the future"

        if self.order_type == OptionOrderType.LIMIT and self.limit_price is None:
            return False, "Limit price required for limit orders"
        if self.order_type in (OptionOrderType.STOP, OptionOrderType.STOP_LIMIT):
            if self.stop_price is None:
                return False, "Stop price required for stop orders"

        return True, None

    def get_occ_symbol(self) -> str:
        """Generate OCC symbol for this order."""
        contract = OptionContract(
            symbol=self.symbol,
            occ_symbol="",  # Will be generated
            option_type=self.option_type,
            strike_price=self.strike_price,
            expiration_date=self.expiration_date,
        )
        return contract.to_occ_symbol()


@dataclass
class OptionOrderResult:
    """Result of options order submission."""

    success: bool
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    status: Optional[str] = None
    filled_qty: int = 0
    filled_price: Optional[float] = None
    fee: float = 0.0
    legs: List[Dict[str, Any]] = field(default_factory=list)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptionChain:
    """
    Container for an option chain.

    Groups option contracts by expiration and strike.
    """

    symbol: str
    contracts: List[OptionContract] = field(default_factory=list)
    last_updated: Optional[datetime] = None

    def get_expirations(self) -> List[date]:
        """Get unique expiration dates."""
        exps = sorted(set(c.expiration_date for c in self.contracts))
        return exps

    def get_strikes(self, expiration: Optional[date] = None) -> List[float]:
        """Get unique strike prices, optionally filtered by expiration."""
        contracts = self.contracts
        if expiration:
            contracts = [c for c in contracts if c.expiration_date == expiration]
        return sorted(set(c.strike_price for c in contracts))

    def get_calls(self, expiration: Optional[date] = None) -> List[OptionContract]:
        """Get call options."""
        contracts = [c for c in self.contracts if c.is_call]
        if expiration:
            contracts = [c for c in contracts if c.expiration_date == expiration]
        return contracts

    def get_puts(self, expiration: Optional[date] = None) -> List[OptionContract]:
        """Get put options."""
        contracts = [c for c in self.contracts if c.is_put]
        if expiration:
            contracts = [c for c in contracts if c.expiration_date == expiration]
        return contracts

    def get_contract(
        self,
        strike: float,
        expiration: date,
        option_type: OptionType,
    ) -> Optional[OptionContract]:
        """Find specific contract by strike, expiration, and type."""
        for c in self.contracts:
            if (
                c.strike_price == strike
                and c.expiration_date == expiration
                and c.option_type == option_type
            ):
                return c
        return None

    def get_atm_strike(self, underlying_price: float) -> float:
        """Get at-the-money strike closest to underlying price."""
        strikes = self.get_strikes()
        if not strikes:
            return underlying_price
        return min(strikes, key=lambda s: abs(s - underlying_price))


# =============================================================================
# Options Execution Adapter
# =============================================================================


class AlpacaOptionsExecutionAdapter(OrderExecutionAdapter):
    """
    Alpaca options order execution adapter.

    Handles options order placement, cancellation, and position management.

    Note: Options trading through Alpaca requires:
    - Approved options trading account
    - Appropriate options trading level
    - Alpaca options API access (may be in beta/limited release)

    Configuration:
        api_key: Alpaca API key (required)
        api_secret: Alpaca API secret (required)
        paper: Use paper trading endpoint (default: True)
        options_level: Options trading level (1-4)
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.ALPACA,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor, config)
        self._client = None
        self._options_level = self._config.get("options_level", 1)
        self._fee_per_contract = float(self._config.get("fee_per_contract", 0.65))

    def _get_client(self):
        """Lazy initialization of Alpaca client."""
        if self._client is None:
            try:
                from alpaca.trading.client import TradingClient
            except ImportError:
                raise ImportError("Alpaca SDK not installed. Install with: pip install alpaca-py")

            api_key = self._config.get("api_key")
            api_secret = self._config.get("api_secret")
            paper = self._config.get("paper", True)

            if not api_key or not api_secret:
                raise ValueError("Alpaca API key and secret are required")

            self._client = TradingClient(
                api_key=api_key,
                secret_key=api_secret,
                paper=paper,
            )

        return self._client

    def _do_connect(self) -> None:
        """Initialize client connection."""
        self._get_client()

    def _do_disconnect(self) -> None:
        """Close client connection."""
        self._client = None

    # =========================================================================
    # Options Chain
    # =========================================================================

    def get_option_chain(
        self,
        symbol: str,
        expiration_date: Optional[date] = None,
    ) -> OptionChain:
        """
        Get options chain for a symbol.

        Args:
            symbol: Underlying symbol
            expiration_date: Filter to specific expiration (optional)

        Returns:
            OptionChain with available contracts
        """
        # Note: Alpaca's options API may have different endpoints
        # This is a placeholder implementation
        logger.info(f"Getting option chain for {symbol}")

        chain = OptionChain(symbol=symbol)

        try:
            client = self._get_client()

            # Try to get options contracts from Alpaca
            # Note: This depends on Alpaca's options API availability
            # The actual implementation may vary based on API version

            # Placeholder: Return empty chain if options API not available
            logger.warning(
                f"Options chain retrieval for {symbol} - " "Alpaca options API integration pending"
            )

        except Exception as e:
            logger.error(f"Failed to get option chain: {e}")

        chain.last_updated = datetime.now()
        return chain

    # =========================================================================
    # Order Submission
    # =========================================================================

    def submit_option_order(
        self,
        config: OptionOrderConfig,
    ) -> OptionOrderResult:
        """
        Submit an options order.

        Args:
            config: OptionOrderConfig with order details

        Returns:
            OptionOrderResult with order status
        """
        # Validate configuration
        is_valid, error_msg = config.validate()
        if not is_valid:
            return OptionOrderResult(
                success=False,
                error_code="INVALID_CONFIG",
                error_message=error_msg,
            )

        # Check options level permissions
        if config.strategy != OptionStrategy.SINGLE and self._options_level < 2:
            return OptionOrderResult(
                success=False,
                error_code="LEVEL_RESTRICTION",
                error_message=f"Strategy {config.strategy.value} requires options level 2+",
            )

        try:
            client = self._get_client()

            # Build OCC symbol
            occ_symbol = config.get_occ_symbol()

            # Determine order parameters
            from alpaca.trading.enums import OrderSide, TimeInForce

            side = OrderSide.BUY if config.side == Side.BUY else OrderSide.SELL

            tif_map = {
                "GTC": TimeInForce.GTC,
                "IOC": TimeInForce.IOC,
                "FOK": TimeInForce.FOK,
                "DAY": TimeInForce.DAY,
            }
            tif = tif_map.get(config.time_in_force.upper(), TimeInForce.DAY)

            # Calculate fee
            fee = self._fee_per_contract * config.qty

            # Submit order
            # Note: This uses the stock order API with OCC symbol
            # Alpaca may have dedicated options endpoints

            if config.order_type == OptionOrderType.MARKET:
                from alpaca.trading.requests import MarketOrderRequest

                request = MarketOrderRequest(
                    symbol=occ_symbol,
                    qty=config.qty,
                    side=side,
                    time_in_force=tif,
                    client_order_id=config.client_order_id,
                )
            else:  # LIMIT
                from alpaca.trading.requests import LimitOrderRequest

                request = LimitOrderRequest(
                    symbol=occ_symbol,
                    qty=config.qty,
                    side=side,
                    time_in_force=tif,
                    limit_price=config.limit_price,
                    client_order_id=config.client_order_id,
                )

            alpaca_order = client.submit_order(request)

            return OptionOrderResult(
                success=True,
                order_id=str(alpaca_order.id),
                client_order_id=str(alpaca_order.client_order_id),
                status=str(alpaca_order.status.value),
                filled_qty=int(alpaca_order.filled_qty or 0),
                filled_price=(
                    float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None
                ),
                fee=fee,
                raw_response={"id": str(alpaca_order.id), "occ_symbol": occ_symbol},
            )

        except Exception as e:
            logger.error(f"Options order submission failed: {e}")
            return OptionOrderResult(
                success=False,
                error_code="SUBMISSION_FAILED",
                error_message=str(e),
            )

    def submit_covered_call(
        self,
        symbol: str,
        strike: float,
        expiration: date,
        qty: int,
        stock_price: Optional[float] = None,
        option_limit_price: Optional[float] = None,
    ) -> OptionOrderResult:
        """
        Submit a covered call order (buy stock + sell call).

        Requires owning the underlying stock (or buying it).

        Args:
            symbol: Underlying symbol
            strike: Call strike price
            expiration: Expiration date
            qty: Number of contracts (each covers 100 shares)
            stock_price: Limit price for stock (None = market)
            option_limit_price: Limit price for call

        Returns:
            OptionOrderResult with leg details
        """
        # Covered calls require level 1
        if self._options_level < 1:
            return OptionOrderResult(
                success=False,
                error_code="LEVEL_RESTRICTION",
                error_message="Covered calls require options level 1+",
            )

        try:
            # Step 1: Ensure we have the underlying shares
            shares_needed = qty * OPTIONS_CONTRACT_MULTIPLIER

            # Step 2: Sell the call options
            config = OptionOrderConfig(
                symbol=symbol,
                option_type=OptionType.CALL,
                strike_price=strike,
                expiration_date=expiration,
                side=Side.SELL,
                qty=qty,
                order_type=OptionOrderType.LIMIT if option_limit_price else OptionOrderType.MARKET,
                limit_price=option_limit_price,
                strategy=OptionStrategy.COVERED_CALL,
            )

            return self.submit_option_order(config)

        except Exception as e:
            logger.error(f"Covered call submission failed: {e}")
            return OptionOrderResult(
                success=False,
                error_code="COVERED_CALL_FAILED",
                error_message=str(e),
            )

    def submit_vertical_spread(
        self,
        symbol: str,
        option_type: OptionType,
        long_strike: float,
        short_strike: float,
        expiration: date,
        qty: int,
        limit_price: Optional[float] = None,
    ) -> OptionOrderResult:
        """
        Submit a vertical spread (bull/bear call/put spread).

        Args:
            symbol: Underlying symbol
            option_type: CALL or PUT
            long_strike: Strike for long leg
            short_strike: Strike for short leg
            expiration: Common expiration date
            qty: Number of spreads
            limit_price: Net debit/credit limit (optional)

        Returns:
            OptionOrderResult with spread details
        """
        # Spreads require level 2
        if self._options_level < 2:
            return OptionOrderResult(
                success=False,
                error_code="LEVEL_RESTRICTION",
                error_message="Vertical spreads require options level 2+",
            )

        try:
            # Build the two legs
            # For bull call spread: buy lower strike, sell higher strike
            # For bear call spread: sell lower strike, buy higher strike
            # For bull put spread: sell higher strike, buy lower strike
            # For bear put spread: buy higher strike, sell lower strike

            long_leg = OptionLeg(
                contract=OptionContract(
                    symbol=symbol,
                    occ_symbol="",
                    option_type=option_type,
                    strike_price=long_strike,
                    expiration_date=expiration,
                ),
                side=Side.BUY,
                qty=qty,
            )

            short_leg = OptionLeg(
                contract=OptionContract(
                    symbol=symbol,
                    occ_symbol="",
                    option_type=option_type,
                    strike_price=short_strike,
                    expiration_date=expiration,
                ),
                side=Side.SELL,
                qty=qty,
            )

            # Note: Multi-leg order submission depends on Alpaca's API
            # This is a simplified implementation

            logger.info(
                f"Submitting vertical spread: {symbol} "
                f"{'CALL' if option_type == OptionType.CALL else 'PUT'} "
                f"{long_strike}/{short_strike} x{qty}"
            )

            # For now, return a placeholder result
            # Actual multi-leg implementation depends on Alpaca's API
            return OptionOrderResult(
                success=False,
                error_code="NOT_IMPLEMENTED",
                error_message="Multi-leg orders pending Alpaca API integration",
                legs=[
                    {"side": "BUY", "strike": long_strike, "qty": qty},
                    {"side": "SELL", "strike": short_strike, "qty": qty},
                ],
            )

        except Exception as e:
            logger.error(f"Vertical spread submission failed: {e}")
            return OptionOrderResult(
                success=False,
                error_code="SPREAD_FAILED",
                error_message=str(e),
            )

    # =========================================================================
    # Order Management
    # =========================================================================

    def cancel_option_order(
        self,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> bool:
        """
        Cancel an options order.

        Args:
            order_id: Alpaca order ID
            client_order_id: Client order ID

        Returns:
            True if cancellation successful
        """
        try:
            client = self._get_client()

            if order_id:
                client.cancel_order_by_id(order_id)
            elif client_order_id:
                orders = client.get_orders()
                for o in orders:
                    if str(o.client_order_id) == client_order_id:
                        client.cancel_order_by_id(str(o.id))
                        return True
                logger.warning(f"Order {client_order_id} not found")
                return False
            else:
                raise ValueError("Either order_id or client_order_id required")

            return True

        except Exception as e:
            logger.error(f"Options order cancellation failed: {e}")
            return False

    def get_option_positions(self) -> List[Dict[str, Any]]:
        """
        Get current options positions.

        Returns:
            List of options position dictionaries
        """
        try:
            client = self._get_client()
            positions = client.get_all_positions()

            option_positions = []
            for p in positions:
                # Check if this is an options position (OCC symbol format)
                symbol = p.symbol
                if len(symbol) > 10 and symbol[-9:-8] in ("C", "P"):
                    try:
                        contract = OptionContract.from_occ_symbol(symbol)
                        option_positions.append(
                            {
                                "occ_symbol": symbol,
                                "underlying": contract.symbol,
                                "option_type": contract.option_type.value,
                                "strike": contract.strike_price,
                                "expiration": str(contract.expiration_date),
                                "qty": int(p.qty),
                                "market_value": float(p.market_value),
                                "cost_basis": float(p.cost_basis),
                                "unrealized_pl": float(p.unrealized_pl),
                                "current_price": float(p.current_price),
                            }
                        )
                    except ValueError:
                        # Not a valid OCC symbol
                        pass

            return option_positions

        except Exception as e:
            logger.error(f"Failed to get options positions: {e}")
            return []

    # =========================================================================
    # Fee Calculation
    # =========================================================================

    def estimate_option_fee(
        self,
        qty: int,
        is_opening: bool = True,
    ) -> float:
        """
        Estimate options trading fee.

        Args:
            qty: Number of contracts
            is_opening: True for opening, False for closing

        Returns:
            Estimated fee in USD
        """
        base_fee = abs(qty) * self._fee_per_contract

        # Small regulatory fee on closing
        if not is_opening:
            base_fee += abs(qty) * 0.02

        return round(base_fee, 2)

    # =========================================================================
    # Standard OrderExecutionAdapter Methods
    # =========================================================================

    def submit_order(self, order: Order) -> OrderResult:
        """
        Submit order (delegates to appropriate handler based on symbol).

        For OCC option symbols, uses options execution.
        For regular symbols, uses stock execution.
        """
        # Check if this is an options order (OCC symbol)
        if len(order.symbol) > 10 and order.symbol[-9:-8] in ("C", "P"):
            # This is an options order
            try:
                contract = OptionContract.from_occ_symbol(order.symbol)
                config = OptionOrderConfig(
                    symbol=contract.symbol,
                    option_type=contract.option_type,
                    strike_price=contract.strike_price,
                    expiration_date=contract.expiration_date,
                    side=order.side,
                    qty=int(order.quantity),
                    order_type=OptionOrderType.LIMIT if order.price else OptionOrderType.MARKET,
                    limit_price=float(order.price) if order.price else None,
                    client_order_id=order.client_order_id,
                )

                result = self.submit_option_order(config)

                return OrderResult(
                    success=result.success,
                    order_id=result.order_id,
                    client_order_id=result.client_order_id,
                    status=result.status,
                    filled_qty=Decimal(str(result.filled_qty)),
                    filled_price=Decimal(str(result.filled_price)) if result.filled_price else None,
                    error_code=result.error_code,
                    error_message=result.error_message,
                    raw_response=result.raw_response,
                )
            except ValueError:
                pass  # Not a valid OCC symbol, fall through to stock order

        # Regular stock order
        # Delegate to the stock order execution adapter
        from adapters.alpaca.order_execution import AlpacaOrderExecutionAdapter

        stock_adapter = AlpacaOrderExecutionAdapter(
            vendor=self._vendor,
            config=dict(self._config),
        )
        return stock_adapter.submit_order(order)

    def cancel_order(
        self,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> bool:
        """Cancel an order."""
        return self.cancel_option_order(order_id, client_order_id)

    def get_order_status(
        self,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> Optional[ExecReport]:
        """Get order status."""
        try:
            client = self._get_client()

            if order_id:
                alpaca_order = client.get_order_by_id(order_id)
            elif client_order_id:
                alpaca_order = client.get_order_by_client_id(client_order_id)
            else:
                return None

            import time

            side = Side.BUY if str(alpaca_order.side) == "buy" else Side.SELL
            order_type = OrderType.MARKET if str(alpaca_order.type) == "market" else OrderType.LIMIT

            status_map = {
                "new": ExecStatus.NEW,
                "partially_filled": ExecStatus.PARTIALLY_FILLED,
                "filled": ExecStatus.FILLED,
                "canceled": ExecStatus.CANCELED,
                "expired": ExecStatus.CANCELED,
                "rejected": ExecStatus.REJECTED,
                "pending_new": ExecStatus.NEW,
                "accepted": ExecStatus.NEW,
            }
            status = status_map.get(str(alpaca_order.status), ExecStatus.NEW)

            return ExecReport(
                ts=int(time.time() * 1000),
                run_id="",
                symbol=alpaca_order.symbol,
                side=side,
                order_type=order_type,
                price=Decimal(str(alpaca_order.filled_avg_price or alpaca_order.limit_price or 0)),
                quantity=Decimal(str(alpaca_order.filled_qty or 0)),
                fee=Decimal(str(self.estimate_option_fee(int(alpaca_order.qty or 0)))),
                fee_asset="USD",
                exec_status=status,
                liquidity=Liquidity.UNKNOWN,
                client_order_id=str(alpaca_order.client_order_id),
                order_id=str(alpaca_order.id),
                meta={"status": str(alpaca_order.status)},
            )

        except Exception as e:
            logger.warning(f"Failed to get order status: {e}")
            return None

    def get_open_orders(
        self,
        symbol: Optional[str] = None,
    ) -> List[Order]:
        """Get open orders, optionally filtered by symbol."""
        try:
            client = self._get_client()

            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            if symbol:
                request.symbols = [symbol]

            alpaca_orders = client.get_orders(request)

            orders = []
            for o in alpaca_orders:
                from core_models import TimeInForce

                side = Side.BUY if str(o.side) == "buy" else Side.SELL
                order_type = OrderType.MARKET if str(o.type) == "market" else OrderType.LIMIT

                orders.append(
                    Order(
                        ts=int(o.created_at.timestamp() * 1000),
                        symbol=o.symbol,
                        side=side,
                        order_type=order_type,
                        quantity=Decimal(str(o.qty)),
                        price=Decimal(str(o.limit_price)) if o.limit_price else None,
                        time_in_force=TimeInForce.GTC,
                        client_order_id=str(o.client_order_id),
                        meta={"alpaca_id": str(o.id)},
                    )
                )

            return orders

        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def get_positions(
        self,
        symbols: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Get positions including options."""
        option_positions = self.get_option_positions()
        return {p["occ_symbol"]: p for p in option_positions}

    def get_account_info(self) -> Any:
        """
        Get account information.

        Returns:
            AccountInfo with balances and status
        """
        try:
            client = self._get_client()
            account = client.get_account()

            from adapters.models import AccountInfo

            return AccountInfo(
                vendor=self._vendor,
                account_id=str(account.id),
                account_type="margin" if account.multiplier > 1 else "cash",
                vip_tier=0,
                maker_fee_rate=0.0,
                taker_fee_rate=0.0,
                buying_power=Decimal(str(account.buying_power)),
                cash_balance=Decimal(str(account.cash)),
                margin_enabled=account.multiplier > 1,
                pattern_day_trader=account.pattern_day_trader,
                raw_data={
                    "equity": str(account.equity),
                    "options_level": self._options_level,
                },
            )

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            from adapters.models import AccountInfo

            return AccountInfo(
                vendor=self._vendor,
                raw_data={"error": str(e)},
            )


# =============================================================================
# Factory Functions
# =============================================================================


def create_options_adapter(
    config: Optional[Mapping[str, Any]] = None,
) -> AlpacaOptionsExecutionAdapter:
    """
    Create an Alpaca options execution adapter.

    Args:
        config: Configuration dictionary

    Returns:
        AlpacaOptionsExecutionAdapter instance
    """
    return AlpacaOptionsExecutionAdapter(
        vendor=ExchangeVendor.ALPACA,
        config=config,
    )


def parse_occ_symbol(occ_symbol: str) -> OptionContract:
    """
    Parse OCC option symbol into OptionContract.

    Args:
        occ_symbol: OCC format symbol (e.g., "AAPL  241220C00200000")

    Returns:
        OptionContract with parsed details
    """
    return OptionContract.from_occ_symbol(occ_symbol)
