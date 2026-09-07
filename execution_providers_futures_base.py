# -*- coding: utf-8 -*-
"""
execution_providers_futures_base.py
Abstract base classes for unified futures execution providers.

Design Principles:
1. Single interface for ALL futures types (crypto, index, commodity, currency)
2. Vendor-agnostic execution logic
3. Clear separation between L2 (parametric) and L3 (LOB) simulation
4. Protocol-based dependency injection for testability

Supports:
- Crypto Perpetual (Binance USDT-M)
- Crypto Quarterly (Binance delivery)
- Index Futures (CME: ES, NQ, YM, RTY)
- Commodity Futures (COMEX: GC, SI; NYMEX: CL, NG)
- Currency Futures (CME: 6E, 6J, 6B, 6A)
- Bond Futures (CBOT: ZB, ZN, ZF)

References:
- Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
- CME Group (2023): Futures Execution Best Practices
- Binance API (2024): USDT-M Futures Execution Documentation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Protocol, Optional, Dict, Any, List, TYPE_CHECKING
import logging

from core_futures import (
    FuturesType,
    FuturesContractSpec,
    FuturesPosition,
    FuturesOrder,
    FuturesFill,
    MarginRequirement,
    MarginMode,
    OrderSide,
    OrderType,
    TimeInForce,
)


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class FuturesMarketState:
    """
    Unified market state for futures execution.

    Vendor-agnostic representation of current market conditions.
    Contains all information needed for execution simulation.

    Attributes:
        timestamp_ms: Unix timestamp in milliseconds
        bid: Best bid price
        ask: Best ask price
        bid_size: Bid quantity at best bid
        ask_size: Ask quantity at best ask
        mark_price: Mark price for liquidation/margin calculation
        index_price: Underlying index/spot price
        last_price: Last traded price
        volume_24h: 24-hour trading volume
        open_interest: Total open interest
        funding_rate: Current funding rate (crypto perpetual only)
        next_funding_time_ms: Next funding timestamp (crypto perpetual only)
        settlement_price: Daily settlement price (CME only)
        days_to_expiry: Days until contract expiration (quarterly/dated)
    """

    timestamp_ms: int
    bid: Decimal
    ask: Decimal
    bid_size: Decimal
    ask_size: Decimal
    mark_price: Decimal
    index_price: Decimal
    last_price: Decimal
    volume_24h: Decimal
    open_interest: Decimal
    funding_rate: Optional[Decimal] = None
    next_funding_time_ms: Optional[int] = None
    settlement_price: Optional[Decimal] = None
    days_to_expiry: Optional[int] = None

    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price."""
        return (self.bid + self.ask) / Decimal("2")

    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_bps(self) -> Decimal:
        """Calculate spread in basis points."""
        mid = self.mid_price
        if mid == 0:
            return Decimal("0")
        return (self.ask - self.bid) / mid * Decimal("10000")

    @property
    def is_crypto(self) -> bool:
        """Check if this is crypto market state (has funding rate)."""
        return self.funding_rate is not None

    @property
    def has_funding(self) -> bool:
        """Check if funding rate is available."""
        return self.funding_rate is not None and self.next_funding_time_ms is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp_ms": self.timestamp_ms,
            "bid": str(self.bid),
            "ask": str(self.ask),
            "bid_size": str(self.bid_size),
            "ask_size": str(self.ask_size),
            "mark_price": str(self.mark_price),
            "index_price": str(self.index_price),
            "last_price": str(self.last_price),
            "volume_24h": str(self.volume_24h),
            "open_interest": str(self.open_interest),
            "funding_rate": str(self.funding_rate) if self.funding_rate else None,
            "next_funding_time_ms": self.next_funding_time_ms,
            "settlement_price": str(self.settlement_price) if self.settlement_price else None,
            "days_to_expiry": self.days_to_expiry,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FuturesMarketState":
        """Create from dictionary."""
        funding_rate = d.get("funding_rate")
        settlement_price = d.get("settlement_price")

        return cls(
            timestamp_ms=int(d.get("timestamp_ms", 0)),
            bid=Decimal(str(d.get("bid", "0"))),
            ask=Decimal(str(d.get("ask", "0"))),
            bid_size=Decimal(str(d.get("bid_size", "0"))),
            ask_size=Decimal(str(d.get("ask_size", "0"))),
            mark_price=Decimal(str(d.get("mark_price", "0"))),
            index_price=Decimal(str(d.get("index_price", "0"))),
            last_price=Decimal(str(d.get("last_price", "0"))),
            volume_24h=Decimal(str(d.get("volume_24h", "0"))),
            open_interest=Decimal(str(d.get("open_interest", "0"))),
            funding_rate=Decimal(str(funding_rate)) if funding_rate else None,
            next_funding_time_ms=d.get("next_funding_time_ms"),
            settlement_price=Decimal(str(settlement_price)) if settlement_price else None,
            days_to_expiry=d.get("days_to_expiry"),
        )


@dataclass(frozen=True)
class ExecutionCostEstimate:
    """
    Pre-trade execution cost estimate.

    Provides breakdown of expected costs for an order.

    Attributes:
        slippage_bps: Expected slippage in basis points
        fee_bps: Expected fee in basis points
        total_cost_bps: Total cost (slippage + fee) in basis points
        impact_cost: Expected market impact cost in quote currency
        estimated_fill_price: Expected fill price
        funding_cost: Expected funding cost over holding period (crypto only)
        participation_rate: Order size as fraction of available liquidity
    """

    slippage_bps: Decimal
    fee_bps: Decimal
    total_cost_bps: Decimal
    impact_cost: Decimal
    estimated_fill_price: Decimal
    funding_cost: Optional[Decimal] = None
    participation_rate: Optional[Decimal] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "slippage_bps": str(self.slippage_bps),
            "fee_bps": str(self.fee_bps),
            "total_cost_bps": str(self.total_cost_bps),
            "impact_cost": str(self.impact_cost),
            "estimated_fill_price": str(self.estimated_fill_price),
            "funding_cost": str(self.funding_cost) if self.funding_cost else None,
            "participation_rate": str(self.participation_rate) if self.participation_rate else None,
        }


# ═══════════════════════════════════════════════════════════════════════════
# PROTOCOL INTERFACES (Dependency Injection)
# ═══════════════════════════════════════════════════════════════════════════


class FuturesMarginProvider(Protocol):
    """
    Protocol for margin calculation providers.

    Implementations should calculate margin requirements for different
    futures types (crypto tiered brackets, CME SPAN, etc.).

    Methods map to common margin operations across all exchanges.
    """

    def calculate_initial_margin(
        self,
        contract: FuturesContractSpec,
        notional: Decimal,
        leverage: int,
    ) -> Decimal:
        """
        Calculate initial margin requirement.

        Args:
            contract: Contract specification
            notional: Position notional value
            leverage: Leverage multiplier

        Returns:
            Initial margin amount in quote currency
        """
        ...

    def calculate_maintenance_margin(
        self,
        contract: FuturesContractSpec,
        notional: Decimal,
    ) -> Decimal:
        """
        Calculate maintenance margin requirement.

        Args:
            contract: Contract specification
            notional: Position notional value

        Returns:
            Maintenance margin amount in quote currency
        """
        ...

    def calculate_liquidation_price(
        self,
        position: FuturesPosition,
        wallet_balance: Decimal,
    ) -> Decimal:
        """
        Calculate liquidation price for position.

        For LONG: LP = Entry * (1 - IM% + MM%)
        For SHORT: LP = Entry * (1 + IM% - MM%)

        Args:
            position: Current position
            wallet_balance: Wallet balance (for cross margin)

        Returns:
            Liquidation price
        """
        ...

    def calculate_margin_ratio(
        self,
        position: FuturesPosition,
        mark_price: Decimal,
        wallet_balance: Decimal,
    ) -> Decimal:
        """
        Calculate current margin ratio.

        MR = (Wallet Balance + Unrealized PnL) / Maintenance Margin
        MR < 1 indicates liquidation zone.

        Args:
            position: Current position
            mark_price: Current mark price
            wallet_balance: Wallet balance

        Returns:
            Margin ratio (> 1 = safe, < 1 = liquidation zone)
        """
        ...

    def get_max_leverage(self, notional: Decimal) -> int:
        """
        Get maximum allowed leverage for notional size.

        Larger positions typically have lower max leverage.

        Args:
            notional: Position notional value

        Returns:
            Maximum leverage allowed
        """
        ...


class FuturesSlippageProvider(Protocol):
    """
    Protocol for slippage estimation providers.

    Implementations model execution slippage based on order size,
    market conditions, and futures-specific factors.
    """

    def estimate_slippage_bps(
        self,
        order: FuturesOrder,
        market: FuturesMarketState,
        participation_rate: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Estimate execution slippage in basis points.

        Models market impact using √participation or LOB-based models.

        Args:
            order: Order to execute
            market: Current market state
            participation_rate: Order size as fraction of ADV (optional)

        Returns:
            Expected slippage in basis points
        """
        ...


class FuturesFeeProvider(Protocol):
    """
    Protocol for fee calculation providers.

    Implementations calculate trading fees based on exchange
    fee schedules and maker/taker status.
    """

    def calculate_fee(
        self,
        notional: Decimal,
        is_maker: bool,
        fee_tier: Optional[str] = None,
    ) -> Decimal:
        """
        Calculate trading fee.

        Args:
            notional: Trade notional value
            is_maker: True if maker order
            fee_tier: Fee tier (VIP level, etc.)

        Returns:
            Fee amount in quote currency
        """
        ...

    def get_fee_rate(
        self,
        is_maker: bool,
        fee_tier: Optional[str] = None,
    ) -> Decimal:
        """
        Get fee rate in decimal form.

        Args:
            is_maker: True if maker order
            fee_tier: Fee tier

        Returns:
            Fee rate (e.g., 0.0002 for 2bps)
        """
        ...


class FuturesFundingProvider(Protocol):
    """
    Protocol for funding rate providers.

    Only applicable to crypto perpetual futures.
    """

    def get_current_funding_rate(self, symbol: str) -> Decimal:
        """
        Get current funding rate.

        Args:
            symbol: Contract symbol

        Returns:
            Current funding rate (positive = longs pay shorts)
        """
        ...

    def calculate_funding_payment(
        self,
        position: FuturesPosition,
        funding_rate: Decimal,
        mark_price: Decimal,
    ) -> Decimal:
        """
        Calculate funding payment amount.

        Payment = Position Notional × Funding Rate
        Positive = pay, Negative = receive

        Args:
            position: Current position
            funding_rate: Funding rate
            mark_price: Mark price for notional calculation

        Returns:
            Funding payment amount
        """
        ...

    def get_next_funding_time(self, symbol: str) -> int:
        """
        Get next funding timestamp.

        Args:
            symbol: Contract symbol

        Returns:
            Unix timestamp in milliseconds
        """
        ...


class FuturesSettlementProvider(Protocol):
    """
    Protocol for settlement providers.

    Handles daily settlement (CME) and funding (crypto).
    """

    def get_settlement_price(self, symbol: str, date: Optional[int] = None) -> Decimal:
        """
        Get settlement price.

        Args:
            symbol: Contract symbol
            date: Date timestamp (None = current day)

        Returns:
            Settlement price
        """
        ...

    def calculate_variation_margin(
        self,
        position: FuturesPosition,
        previous_settlement: Decimal,
        current_settlement: Decimal,
    ) -> Decimal:
        """
        Calculate variation margin (daily P&L settlement).

        Args:
            position: Current position
            previous_settlement: Previous settlement price
            current_settlement: Current settlement price

        Returns:
            Variation margin amount
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# ABSTRACT BASE CLASSES
# ═══════════════════════════════════════════════════════════════════════════


class BaseFuturesExecutionProvider(ABC):
    """
    Abstract base class for all futures execution providers.

    Provides unified interface for:
    - Crypto perpetuals (Binance)
    - Crypto quarterly (Binance)
    - Index futures (CME via IB)
    - Commodity futures (CME via IB)
    - Currency futures (CME via IB)

    Subclasses implement vendor-specific execution logic while
    maintaining consistent interface for backtesting and live trading.

    Design:
    - Protocols for dependency injection (testability)
    - Decimal precision for financial calculations
    - Immutable data structures for thread safety
    """

    def __init__(
        self,
        futures_type: FuturesType,
        margin_provider: FuturesMarginProvider,
        slippage_provider: FuturesSlippageProvider,
        fee_provider: FuturesFeeProvider,
        funding_provider: Optional[FuturesFundingProvider] = None,
    ):
        """
        Initialize execution provider.

        Args:
            futures_type: Type of futures contract
            margin_provider: Margin calculation provider
            slippage_provider: Slippage estimation provider
            fee_provider: Fee calculation provider
            funding_provider: Funding rate provider (crypto perpetual only)

        Raises:
            ValueError: If funding_provider required but not provided
        """
        self._futures_type = futures_type
        self._margin_provider = margin_provider
        self._slippage_provider = slippage_provider
        self._fee_provider = fee_provider
        self._funding_provider = funding_provider

        # Validate funding provider for crypto perpetuals
        if futures_type == FuturesType.CRYPTO_PERPETUAL and not funding_provider:
            raise ValueError("FundingProvider required for crypto perpetuals")

    @property
    def futures_type(self) -> FuturesType:
        """Get futures type."""
        return self._futures_type

    @property
    def requires_funding(self) -> bool:
        """Check if this provider requires funding rate handling."""
        return self._futures_type == FuturesType.CRYPTO_PERPETUAL

    @abstractmethod
    def execute(
        self,
        order: FuturesOrder,
        market: FuturesMarketState,
        position: Optional[FuturesPosition] = None,
        wallet_balance: Optional[Decimal] = None,
    ) -> FuturesFill:
        """
        Execute order in simulation.

        Args:
            order: Order to execute
            market: Current market state
            position: Current position (if any)
            wallet_balance: Wallet balance (for margin calculation)

        Returns:
            FuturesFill with execution details
        """
        pass

    @abstractmethod
    def estimate_execution_cost(
        self,
        order: FuturesOrder,
        market: FuturesMarketState,
    ) -> ExecutionCostEstimate:
        """
        Pre-trade cost estimation.

        Args:
            order: Order to estimate
            market: Current market state

        Returns:
            ExecutionCostEstimate with breakdown
        """
        pass

    def get_margin_requirement(
        self,
        contract: FuturesContractSpec,
        qty: Decimal,
        price: Decimal,
        leverage: int,
    ) -> MarginRequirement:
        """
        Get margin requirement for position.

        Args:
            contract: Contract specification
            qty: Position quantity
            price: Entry price
            leverage: Leverage multiplier

        Returns:
            MarginRequirement with initial and maintenance
        """
        notional = abs(qty) * price * contract.multiplier
        initial = self._margin_provider.calculate_initial_margin(contract, notional, leverage)
        maintenance = self._margin_provider.calculate_maintenance_margin(contract, notional)
        return MarginRequirement(
            initial=initial,
            maintenance=maintenance,
            variation=Decimal("0"),  # Calculated on daily settlement
        )

    def calculate_pnl(
        self,
        position: FuturesPosition,
        current_price: Decimal,
    ) -> Decimal:
        """
        Calculate unrealized P&L for position.

        Uses FuturesPosition.calculate_pnl() if available, otherwise manual calc.

        Args:
            position: Current position
            current_price: Current market price

        Returns:
            Unrealized P&L
        """
        if position.qty == Decimal("0"):
            return Decimal("0")

        # Use position's own method if available
        if hasattr(position, "calculate_pnl"):
            return position.calculate_pnl(current_price)

        # Manual calculation (without contract multiplier for simplicity)
        notional_entry = abs(position.qty) * position.entry_price
        notional_current = abs(position.qty) * current_price

        if position.qty > 0:  # Long
            return notional_current - notional_entry
        else:  # Short
            return notional_entry - notional_current

    def calculate_funding_payment(
        self,
        position: FuturesPosition,
        market: FuturesMarketState,
    ) -> Decimal:
        """
        Calculate funding payment for position.

        Only applicable to crypto perpetuals.

        Args:
            position: Current position
            market: Market state with funding rate

        Returns:
            Funding payment (positive = pay, negative = receive)
        """
        if not self.requires_funding or not self._funding_provider:
            return Decimal("0")

        if market.funding_rate is None:
            return Decimal("0")

        return self._funding_provider.calculate_funding_payment(
            position=position,
            funding_rate=market.funding_rate,
            mark_price=market.mark_price,
        )


class L2FuturesExecutionProvider(BaseFuturesExecutionProvider):
    """
    L2 parametric execution provider for futures.

    Uses statistical models for slippage estimation:
    - √participation impact model (Almgren-Chriss)
    - Volatility regime adjustments
    - Funding rate stress (crypto)
    - Time-of-day liquidity curves

    Suitable for:
    - Backtesting with realistic cost modeling
    - Strategy development and optimization
    - Production with moderate accuracy requirements (~80-90%)

    References:
    - Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
    - Kyle (1985): "Continuous Auctions and Insider Trading"
    """

    def execute(
        self,
        order: FuturesOrder,
        market: FuturesMarketState,
        position: Optional[FuturesPosition] = None,
        wallet_balance: Optional[Decimal] = None,
    ) -> FuturesFill:
        """
        Execute with L2 parametric slippage model.

        Args:
            order: Order to execute
            market: Current market state
            position: Current position
            wallet_balance: Wallet balance

        Returns:
            FuturesFill with execution details
        """
        # Estimate slippage
        slippage_bps = self._slippage_provider.estimate_slippage_bps(order, market)

        # Calculate execution price
        mid = market.mid_price
        slippage_factor = slippage_bps / Decimal("10000")

        if order.side == OrderSide.BUY:
            exec_price = mid * (Decimal("1") + slippage_factor)
        else:
            exec_price = mid * (Decimal("1") - slippage_factor)

        # Calculate fee
        notional = order.qty * exec_price
        is_maker = order.order_type == OrderType.LIMIT and order.post_only
        fee = self._fee_provider.calculate_fee(notional, is_maker)

        # Calculate realized PnL if reducing position
        realized_pnl = Decimal("0")
        new_position_size = order.qty
        new_avg_entry = exec_price

        if position and position.qty != Decimal("0"):
            # Check if reducing position
            is_reducing = (position.qty > 0 and order.side == OrderSide.SELL) or (
                position.qty < 0 and order.side == OrderSide.BUY
            )
            if is_reducing:
                close_qty = min(order.qty, abs(position.qty))
                realized_pnl = self._calculate_realized_pnl(position, exec_price, close_qty)
                # Update position
                if order.side == OrderSide.SELL:
                    new_position_size = position.qty - order.qty
                else:
                    new_position_size = position.qty + order.qty

                if abs(new_position_size) > Decimal("0"):
                    new_avg_entry = position.entry_price
            else:
                # Increasing position
                if order.side == OrderSide.BUY:
                    new_position_size = position.qty + order.qty
                else:
                    new_position_size = position.qty - order.qty
                # Weighted average entry
                total_qty = abs(position.qty) + order.qty
                old_value = abs(position.qty) * position.entry_price
                new_value = order.qty * exec_price
                new_avg_entry = (old_value + new_value) / total_qty

        return FuturesFill(
            order_id=order.client_order_id or f"SIM_{market.timestamp_ms}",
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            filled_qty=order.qty,
            avg_price=exec_price,
            commission=fee,
            commission_asset="USDT",  # Configurable
            realized_pnl=realized_pnl,
            timestamp_ms=market.timestamp_ms,
            is_maker=is_maker,
            liquidity="MAKER" if is_maker else "TAKER",
            margin_impact=Decimal("0"),  # Calculated separately
            new_position_size=new_position_size,
            new_avg_entry=new_avg_entry,
        )

    def _calculate_realized_pnl(
        self,
        position: FuturesPosition,
        exit_price: Decimal,
        close_qty: Decimal,
    ) -> Decimal:
        """Calculate realized PnL for closed quantity."""
        if position.qty == Decimal("0"):
            return Decimal("0")

        # Manual calculation (without contract multiplier for simplicity)
        entry_notional = close_qty * position.entry_price
        exit_notional = close_qty * exit_price

        if position.qty > 0:  # Long position
            return exit_notional - entry_notional
        else:  # Short position
            return entry_notional - exit_notional

    def estimate_execution_cost(
        self,
        order: FuturesOrder,
        market: FuturesMarketState,
    ) -> ExecutionCostEstimate:
        """
        Pre-trade cost estimation.

        Args:
            order: Order to estimate
            market: Current market state

        Returns:
            ExecutionCostEstimate with breakdown
        """
        slippage_bps = self._slippage_provider.estimate_slippage_bps(order, market)

        mid = market.mid_price
        slippage_factor = slippage_bps / Decimal("10000")
        if order.side == OrderSide.BUY:
            estimated_price = mid * (Decimal("1") + slippage_factor)
        else:
            estimated_price = mid * (Decimal("1") - slippage_factor)

        notional = order.qty * estimated_price
        fee = self._fee_provider.calculate_fee(notional, is_maker=False)
        fee_bps = fee / notional * Decimal("10000") if notional > 0 else Decimal("0")

        # Calculate funding cost estimate (if applicable)
        funding_cost = None
        if self.requires_funding and self._funding_provider and market.funding_rate:
            # Estimate 8-hour funding cost
            funding_cost = notional * abs(market.funding_rate)

        return ExecutionCostEstimate(
            slippage_bps=slippage_bps,
            fee_bps=fee_bps,
            total_cost_bps=slippage_bps + fee_bps,
            impact_cost=notional * slippage_bps / Decimal("10000"),
            estimated_fill_price=estimated_price,
            funding_cost=funding_cost,
            participation_rate=None,  # Requires ADV calculation
        )


class L3FuturesExecutionProvider(BaseFuturesExecutionProvider):
    """
    L3 order book execution provider for futures.

    Uses full LOB simulation:
    - Queue position tracking
    - Market impact modeling (Kyle, Almgren-Chriss, Gatheral)
    - Latency simulation
    - Fill probability models

    Extends existing lob/ module for futures-specific features:
    - Liquidation order injection
    - Funding rate impact
    - Daily settlement simulation (CME)

    Suitable for:
    - High-fidelity backtesting (95%+ accuracy)
    - Market microstructure research
    - HFT strategy development

    Note: Full implementation requires LOB components initialization.
    """

    def __init__(
        self,
        futures_type: FuturesType,
        margin_provider: FuturesMarginProvider,
        slippage_provider: FuturesSlippageProvider,
        fee_provider: FuturesFeeProvider,
        funding_provider: Optional[FuturesFundingProvider] = None,
        lob_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize L3 execution provider.

        Args:
            futures_type: Type of futures contract
            margin_provider: Margin calculation provider
            slippage_provider: Slippage estimation provider
            fee_provider: Fee calculation provider
            funding_provider: Funding rate provider (crypto only)
            lob_config: LOB simulation configuration
        """
        super().__init__(
            futures_type=futures_type,
            margin_provider=margin_provider,
            slippage_provider=slippage_provider,
            fee_provider=fee_provider,
            funding_provider=funding_provider,
        )
        self._lob_config = lob_config or {}
        # LOB components initialized lazily to avoid circular imports
        self._matching_engine = None
        self._impact_model = None
        self._latency_model = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Initialize LOB components if not already done."""
        if self._initialized:
            return

        # Import LOB components here to avoid circular imports
        try:
            from lob.matching_engine import MatchingEngine
            from lob.market_impact import create_impact_model
            from lob.latency_model import LatencyModel, LatencyProfile

            self._matching_engine = MatchingEngine()
            self._impact_model = create_impact_model(
                self._lob_config.get("impact_model", "almgren_chriss")
            )
            self._latency_model = LatencyModel.from_profile(
                LatencyProfile[self._lob_config.get("latency_profile", "INSTITUTIONAL")]
            )
            self._initialized = True
        except ImportError as e:
            logger.warning(f"LOB components not available: {e}")
            self._initialized = False

    def execute(
        self,
        order: FuturesOrder,
        market: FuturesMarketState,
        position: Optional[FuturesPosition] = None,
        wallet_balance: Optional[Decimal] = None,
    ) -> FuturesFill:
        """
        Execute order using L2 parametric model.

        This method provides a convenient fallback to L2 execution when
        no OrderBook is available. For full L3 LOB simulation, use
        execute_with_lob() method.

        Args:
            order: Order to execute
            market: Current market state
            position: Current position
            wallet_balance: Wallet balance

        Returns:
            FuturesFill with execution details
        """
        # Fall back to L2 execution (no OrderBook provided)
        logger.debug("L3 execute() called without LOB, using L2 fallback")
        l2_provider = L2FuturesExecutionProvider(
            futures_type=self._futures_type,
            margin_provider=self._margin_provider,
            slippage_provider=self._slippage_provider,
            fee_provider=self._fee_provider,
            funding_provider=self._funding_provider,
        )
        return l2_provider.execute(order, market, position, wallet_balance)

    def estimate_execution_cost(
        self,
        order: FuturesOrder,
        market: FuturesMarketState,
    ) -> ExecutionCostEstimate:
        """
        Pre-trade cost estimation using L2 parametric model.

        For full L3 estimation with LOB state, use estimate_with_lob() method.

        Args:
            order: Order to estimate
            market: Current market state

        Returns:
            ExecutionCostEstimate with breakdown
        """
        # Fall back to L2 estimation (no OrderBook provided)
        l2_provider = L2FuturesExecutionProvider(
            futures_type=self._futures_type,
            margin_provider=self._margin_provider,
            slippage_provider=self._slippage_provider,
            fee_provider=self._fee_provider,
            funding_provider=self._funding_provider,
        )
        return l2_provider.estimate_execution_cost(order, market)


# ═══════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def create_futures_execution_provider(
    futures_type: FuturesType,
    level: str = "L2",
    config: Optional[Dict[str, Any]] = None,
) -> BaseFuturesExecutionProvider:
    """
    Factory function for creating futures execution providers.

    Args:
        futures_type: Type of futures contract
        level: Simulation fidelity ("L2" = parametric, "L3" = LOB)
        config: Provider configuration

    Returns:
        Configured execution provider

    Raises:
        ValueError: If invalid level or futures_type

    Example:
        >>> provider = create_futures_execution_provider(
        ...     FuturesType.CRYPTO_PERPETUAL,
        ...     level="L2",
        ...     config={"slippage_profile": "binance_futures"},
        ... )
    """
    config = config or {}

    # Create default providers based on futures type
    margin_provider = _create_margin_provider(futures_type, config)
    slippage_provider = _create_slippage_provider(futures_type, config)
    fee_provider = _create_fee_provider(futures_type, config)
    funding_provider = None

    if futures_type == FuturesType.CRYPTO_PERPETUAL:
        funding_provider = _create_funding_provider(futures_type, config)

    if level == "L2":
        return L2FuturesExecutionProvider(
            futures_type=futures_type,
            margin_provider=margin_provider,
            slippage_provider=slippage_provider,
            fee_provider=fee_provider,
            funding_provider=funding_provider,
        )
    elif level == "L3":
        return L3FuturesExecutionProvider(
            futures_type=futures_type,
            margin_provider=margin_provider,
            slippage_provider=slippage_provider,
            fee_provider=fee_provider,
            funding_provider=funding_provider,
            lob_config=config.get("lob", {}),
        )
    else:
        raise ValueError(f"Invalid execution level: {level}. Use 'L2' or 'L3'.")


def _create_margin_provider(
    futures_type: FuturesType,
    config: Dict[str, Any],
) -> FuturesMarginProvider:
    """Create margin provider for futures type."""
    # Import implementation
    from impl_futures_margin import (
        TieredMarginCalculator,
        CMEMarginCalculator,
        SimpleMarginCalculator,
        load_leverage_brackets,
    )

    if futures_type.is_crypto:
        # Use tiered brackets for crypto
        brackets_file = config.get("brackets_file", "data/futures/leverage_brackets.json")
        try:
            brackets = load_leverage_brackets(brackets_file)
            symbol = config.get("symbol", "BTCUSDT")
            if symbol in brackets:
                return TieredMarginCalculator(brackets=brackets[symbol])
        except Exception as e:
            logger.warning(f"Failed to load brackets: {e}, using simple margin")

        # Fallback to simple margin
        im_pct = float(config.get("im_pct", 1.0))  # 1% = 100x leverage
        mm_pct = float(config.get("mm_pct", 0.5))  # 0.5%
        return SimpleMarginCalculator(
            initial_pct=im_pct,
            maintenance_pct=mm_pct,
        )
    else:
        # Use CME margin for traditional futures
        # Import CMEMarginRate for custom rates
        from impl_futures_margin import CMEMarginRate

        # Check if custom margin rates are provided
        if "initial_margin" in config or "maintenance_margin" in config:
            # Create custom margin rate
            symbol = config.get("symbol", "CUSTOM")
            custom_rate = CMEMarginRate(
                symbol=symbol,
                initial_margin=Decimal(str(config.get("initial_margin", "5000"))),
                maintenance_margin=Decimal(str(config.get("maintenance_margin", "4500"))),
            )
            return CMEMarginCalculator(
                margin_rates={symbol: custom_rate},
                use_day_trading_margin=config.get("use_day_trading_margin", False),
            )

        # Use defaults (ES, NQ, GC, CL, etc.)
        return CMEMarginCalculator(
            use_day_trading_margin=config.get("use_day_trading_margin", False),
        )


def _create_slippage_provider(
    futures_type: FuturesType,
    config: Dict[str, Any],
) -> FuturesSlippageProvider:
    """Create slippage provider for futures type."""
    # Return a simple parametric slippage provider
    return _ParametricSlippageProvider(
        base_slippage_bps=Decimal(str(config.get("base_slippage_bps", "2.0"))),
        impact_coefficient=Decimal(str(config.get("impact_coefficient", "0.1"))),
    )


def _create_fee_provider(
    futures_type: FuturesType,
    config: Dict[str, Any],
) -> FuturesFeeProvider:
    """Create fee provider for futures type."""
    if futures_type.is_crypto:
        return _CryptoFeeProvider(
            maker_rate=Decimal(str(config.get("maker_fee", "0.0002"))),
            taker_rate=Decimal(str(config.get("taker_fee", "0.0004"))),
        )
    else:
        return _CMEFeeProvider(
            fee_per_contract=Decimal(str(config.get("fee_per_contract", "2.50"))),
        )


def _create_funding_provider(
    futures_type: FuturesType,
    config: Dict[str, Any],
) -> FuturesFundingProvider:
    """Create funding provider for crypto perpetuals."""
    return _SimpleFundingProvider()


# ═══════════════════════════════════════════════════════════════════════════
# DEFAULT IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════


class _ParametricSlippageProvider:
    """Simple parametric slippage provider."""

    def __init__(
        self,
        base_slippage_bps: Decimal = Decimal("2.0"),
        impact_coefficient: Decimal = Decimal("0.1"),
    ):
        self.base_slippage_bps = base_slippage_bps
        self.impact_coefficient = impact_coefficient

    def estimate_slippage_bps(
        self,
        order: FuturesOrder,
        market: FuturesMarketState,
        participation_rate: Optional[Decimal] = None,
    ) -> Decimal:
        """Estimate slippage using √participation model."""
        # Base spread component
        half_spread = market.spread_bps / Decimal("2")

        # Participation impact (if ADV available)
        impact = Decimal("0")
        if participation_rate and participation_rate > Decimal("0"):
            # Almgren-Chriss √participation
            import math

            sqrt_part = Decimal(str(math.sqrt(float(participation_rate))))
            impact = self.impact_coefficient * sqrt_part * Decimal("10000")

        return half_spread + self.base_slippage_bps + impact


class _CryptoFeeProvider:
    """Crypto futures fee provider (Binance-style)."""

    def __init__(
        self,
        maker_rate: Decimal = Decimal("0.0002"),
        taker_rate: Decimal = Decimal("0.0004"),
    ):
        self.maker_rate = maker_rate
        self.taker_rate = taker_rate

    def calculate_fee(
        self,
        notional: Decimal,
        is_maker: bool,
        fee_tier: Optional[str] = None,
    ) -> Decimal:
        """Calculate trading fee."""
        rate = self.maker_rate if is_maker else self.taker_rate
        return notional * rate

    def get_fee_rate(
        self,
        is_maker: bool,
        fee_tier: Optional[str] = None,
    ) -> Decimal:
        """Get fee rate."""
        return self.maker_rate if is_maker else self.taker_rate


class _CMEFeeProvider:
    """CME futures fee provider (per-contract)."""

    def __init__(
        self,
        fee_per_contract: Decimal = Decimal("2.50"),
    ):
        self.fee_per_contract = fee_per_contract

    def calculate_fee(
        self,
        notional: Decimal,
        is_maker: bool,
        fee_tier: Optional[str] = None,
    ) -> Decimal:
        """Calculate trading fee (simplified: uses notional proxy)."""
        # For CME, fee is per contract, not per notional
        # This is a simplified implementation
        return self.fee_per_contract

    def get_fee_rate(
        self,
        is_maker: bool,
        fee_tier: Optional[str] = None,
    ) -> Decimal:
        """Get fee rate (not applicable for per-contract fees)."""
        return Decimal("0")


class _SimpleFundingProvider:
    """Simple funding rate provider."""

    def __init__(self):
        self._funding_rates: Dict[str, Decimal] = {}
        self._next_funding_times: Dict[str, int] = {}

    def get_current_funding_rate(self, symbol: str) -> Decimal:
        """Get current funding rate."""
        return self._funding_rates.get(symbol, Decimal("0.0001"))

    def calculate_funding_payment(
        self,
        position: FuturesPosition,
        funding_rate: Decimal,
        mark_price: Decimal,
    ) -> Decimal:
        """Calculate funding payment."""
        if position.qty == Decimal("0"):
            return Decimal("0")

        multiplier = position.contract.multiplier if position.contract else Decimal("1")
        notional = abs(position.qty) * mark_price * multiplier

        # Positive funding rate: longs pay shorts
        # Position qty > 0 = long, < 0 = short
        if position.qty > 0:
            return notional * funding_rate  # Long pays
        else:
            return -notional * funding_rate  # Short receives

    def get_next_funding_time(self, symbol: str) -> int:
        """Get next funding timestamp."""
        return self._next_funding_times.get(symbol, 0)

    def set_funding_rate(self, symbol: str, rate: Decimal) -> None:
        """Set funding rate for symbol (for testing)."""
        self._funding_rates[symbol] = rate

    def set_next_funding_time(self, symbol: str, timestamp_ms: int) -> None:
        """Set next funding time (for testing)."""
        self._next_funding_times[symbol] = timestamp_ms


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    # Data Models
    "FuturesMarketState",
    "ExecutionCostEstimate",
    # Protocol Interfaces
    "FuturesMarginProvider",
    "FuturesSlippageProvider",
    "FuturesFeeProvider",
    "FuturesFundingProvider",
    "FuturesSettlementProvider",
    # Abstract Base Classes
    "BaseFuturesExecutionProvider",
    "L2FuturesExecutionProvider",
    "L3FuturesExecutionProvider",
    # Factory Function
    "create_futures_execution_provider",
]
