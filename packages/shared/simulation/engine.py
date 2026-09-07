# -*- coding: utf-8 -*-
"""
Simulation Execution Engine.

Processes OrderIntents in simulation mode without real broker connections.
This is safe for Cloud zone deployment.

Key Features:
- Simulates order fills with configurable slippage
- Tracks positions and P&L
- Supports multiple symbols
- Generates simulated telemetry
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Final, List, Optional
from uuid import UUID, uuid4

from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide


class FillStatus(str, Enum):
    """Status of simulated fill."""

    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class SimulationConfig:
    """
    Configuration for simulation engine.
    """

    # Slippage settings
    slippage_bps: Decimal = Decimal("5")  # Default 5 basis points
    use_random_slippage: bool = False
    max_slippage_bps: Decimal = Decimal("50")

    # Fill simulation
    fill_probability: float = 0.98  # 98% fill rate
    partial_fill_probability: float = 0.05  # 5% partial fills
    average_fill_ratio: float = 0.7  # Average 70% fill on partial

    # Latency simulation
    min_latency_ms: int = 1
    max_latency_ms: int = 100

    # Market impact
    enable_market_impact: bool = True
    impact_coefficient: Decimal = Decimal("0.0001")

    # Initial capital
    initial_capital: Decimal = Decimal("100000")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "slippage_bps": str(self.slippage_bps),
            "use_random_slippage": self.use_random_slippage,
            "max_slippage_bps": str(self.max_slippage_bps),
            "fill_probability": self.fill_probability,
            "partial_fill_probability": self.partial_fill_probability,
            "average_fill_ratio": self.average_fill_ratio,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "enable_market_impact": self.enable_market_impact,
            "impact_coefficient": str(self.impact_coefficient),
            "initial_capital": str(self.initial_capital),
        }


@dataclass
class SimulatedFill:
    """
    Represents a simulated order fill.
    """

    fill_id: UUID = field(default_factory=uuid4)
    intent_id: UUID = field(default_factory=uuid4)
    symbol: str = ""
    side: IntentSide = IntentSide.LONG
    quantity: Decimal = Decimal("0")
    fill_price: Decimal = Decimal("0")
    commission: Decimal = Decimal("0")
    slippage: Decimal = Decimal("0")
    status: FillStatus = FillStatus.FILLED
    timestamp: datetime = field(default_factory=datetime.utcnow)
    latency_ms: int = 0

    @property
    def notional(self) -> Decimal:
        """Calculate notional value."""
        return self.quantity * self.fill_price

    @property
    def total_cost(self) -> Decimal:
        """Calculate total cost including commission."""
        return self.notional + self.commission

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fill_id": str(self.fill_id),
            "intent_id": str(self.intent_id),
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": str(self.quantity),
            "fill_price": str(self.fill_price),
            "commission": str(self.commission),
            "slippage": str(self.slippage),
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": self.latency_ms,
            "notional": str(self.notional),
            "total_cost": str(self.total_cost),
        }


@dataclass
class SimulatedPosition:
    """
    Represents a simulated position.
    """

    symbol: str = ""
    quantity: Decimal = Decimal("0")
    avg_price: Decimal = Decimal("0")
    cost_basis: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    side: IntentSide = IntentSide.FLAT
    opened_at: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_flat(self) -> bool:
        """Check if position is flat."""
        return self.quantity == Decimal("0")

    @property
    def market_value(self) -> Decimal:
        """Calculate market value at current price (needs price input)."""
        return self.quantity * self.avg_price

    def update_unrealized_pnl(self, current_price: Decimal) -> None:
        """Update unrealized P&L based on current price."""
        if self.is_flat:
            self.unrealized_pnl = Decimal("0")
            return

        if self.side == IntentSide.LONG:
            self.unrealized_pnl = (current_price - self.avg_price) * self.quantity
        else:
            self.unrealized_pnl = (self.avg_price - current_price) * abs(self.quantity)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "quantity": str(self.quantity),
            "avg_price": str(self.avg_price),
            "cost_basis": str(self.cost_basis),
            "unrealized_pnl": str(self.unrealized_pnl),
            "realized_pnl": str(self.realized_pnl),
            "side": self.side.value,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "last_updated": self.last_updated.isoformat(),
            "is_flat": self.is_flat,
        }


class SimExecutionEngine:
    """
    Simulation Execution Engine.

    Processes OrderIntents and generates simulated fills without
    connecting to real brokers. Safe for Cloud zone.

    Usage:
        engine = SimExecutionEngine(config=SimulationConfig())
        engine.set_price("BTCUSDT", Decimal("50000"))

        intent = OrderIntent(...)
        fill = engine.process_intent(intent)
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """Initialize simulation engine."""
        self.config = config or SimulationConfig()
        self._positions: Dict[str, SimulatedPosition] = {}
        self._fills: List[SimulatedFill] = []
        self._prices: Dict[str, Decimal] = {}
        self._capital = self.config.initial_capital
        self._total_pnl = Decimal("0")

    def set_price(self, symbol: str, price: Decimal) -> None:
        """Set current price for symbol."""
        self._prices[symbol] = price
        # Update unrealized P&L for position
        if symbol in self._positions:
            self._positions[symbol].update_unrealized_pnl(price)

    def get_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price for symbol."""
        return self._prices.get(symbol)

    def get_position(self, symbol: str) -> SimulatedPosition:
        """Get position for symbol (creates if not exists)."""
        if symbol not in self._positions:
            self._positions[symbol] = SimulatedPosition(symbol=symbol)
        return self._positions[symbol]

    def process_intent(self, intent: OrderIntent) -> Optional[SimulatedFill]:
        """
        Process an OrderIntent and generate simulated fill.

        Args:
            intent: The order intent to process

        Returns:
            SimulatedFill if intent results in an order, None for HOLD/NO_ACTION
        """
        # Skip passive intents
        if intent.is_passive:
            return None

        # Get current price
        price = self._prices.get(intent.symbol)
        if price is None:
            # No price available - reject
            return SimulatedFill(
                intent_id=intent.intent_id,
                symbol=intent.symbol,
                status=FillStatus.REJECTED,
            )

        # Determine quantity
        quantity = intent.target_quantity
        if quantity is None and intent.target_notional is not None:
            quantity = intent.target_notional / price

        if quantity is None:
            # Handle flatten all
            if intent.intent_type == IntentType.FLATTEN_ALL:
                position = self.get_position(intent.symbol)
                quantity = abs(position.quantity)
            else:
                return SimulatedFill(
                    intent_id=intent.intent_id,
                    symbol=intent.symbol,
                    status=FillStatus.REJECTED,
                )

        # Calculate fill price with slippage
        slippage_multiplier = Decimal("1") + (self.config.slippage_bps / Decimal("10000"))
        if intent.side == IntentSide.LONG:
            fill_price = price * slippage_multiplier
        else:
            fill_price = price / slippage_multiplier

        # Use limit price if specified and better
        if intent.limit_price is not None:
            if intent.side == IntentSide.LONG:
                fill_price = min(fill_price, intent.limit_price)
            else:
                fill_price = max(fill_price, intent.limit_price)

        # Calculate slippage
        slippage = abs(fill_price - price)

        # Calculate commission (example: 0.1%)
        commission = quantity * fill_price * Decimal("0.001")

        # Create fill
        fill = SimulatedFill(
            intent_id=intent.intent_id,
            symbol=intent.symbol,
            side=intent.side,
            quantity=quantity,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage,
            status=FillStatus.FILLED,
            latency_ms=50,  # Simulated latency
        )

        # Update position
        self._update_position(fill)

        # Record fill
        self._fills.append(fill)

        return fill

    def _update_position(self, fill: SimulatedFill) -> None:
        """Update position based on fill."""
        position = self.get_position(fill.symbol)

        if fill.side == IntentSide.LONG:
            # Buying
            if position.side == IntentSide.SHORT:
                # Closing short
                realized = (position.avg_price - fill.fill_price) * min(
                    fill.quantity, abs(position.quantity)
                )
                position.realized_pnl += realized
                self._total_pnl += realized
                position.quantity += fill.quantity
            else:
                # Opening or adding to long
                total_cost = position.cost_basis + (fill.quantity * fill.fill_price)
                total_qty = position.quantity + fill.quantity
                if total_qty > 0:
                    position.avg_price = total_cost / total_qty
                position.quantity = total_qty
                position.cost_basis = total_cost
                position.side = IntentSide.LONG

        elif fill.side == IntentSide.SHORT:
            # Selling
            if position.side == IntentSide.LONG:
                # Closing long
                realized = (fill.fill_price - position.avg_price) * min(
                    fill.quantity, position.quantity
                )
                position.realized_pnl += realized
                self._total_pnl += realized
                position.quantity -= fill.quantity
            else:
                # Opening or adding to short
                position.quantity -= fill.quantity
                position.side = IntentSide.SHORT

        elif fill.side == IntentSide.FLAT:
            # Flatten
            if not position.is_flat:
                price = self._prices.get(fill.symbol, fill.fill_price)
                if position.side == IntentSide.LONG:
                    realized = (price - position.avg_price) * position.quantity
                else:
                    realized = (position.avg_price - price) * abs(position.quantity)
                position.realized_pnl += realized
                self._total_pnl += realized
                position.quantity = Decimal("0")
                position.side = IntentSide.FLAT

        # Update timestamp
        position.last_updated = datetime.utcnow()
        if position.opened_at is None and not position.is_flat:
            position.opened_at = datetime.utcnow()

    def get_fills(self) -> List[SimulatedFill]:
        """Get all fills."""
        return list(self._fills)

    def get_positions(self) -> Dict[str, SimulatedPosition]:
        """Get all positions."""
        return dict(self._positions)

    def get_total_pnl(self) -> Decimal:
        """Get total realized P&L."""
        return self._total_pnl

    def get_equity(self) -> Decimal:
        """Get current equity (capital + unrealized P&L)."""
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        return self._capital + self._total_pnl + unrealized

    def reset(self) -> None:
        """Reset engine state."""
        self._positions.clear()
        self._fills.clear()
        self._prices.clear()
        self._capital = self.config.initial_capital
        self._total_pnl = Decimal("0")

    def to_dict(self) -> Dict[str, Any]:
        """Get engine state as dictionary."""
        return {
            "capital": str(self._capital),
            "total_pnl": str(self._total_pnl),
            "equity": str(self.get_equity()),
            "positions": {s: p.to_dict() for s, p in self._positions.items()},
            "fills_count": len(self._fills),
            "config": self.config.to_dict(),
        }
