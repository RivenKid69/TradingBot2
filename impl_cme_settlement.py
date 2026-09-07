# -*- coding: utf-8 -*-
"""
impl_cme_settlement.py
CME daily settlement simulation for traditional futures.

IMPORTANT: Settlement times vary by product!
- Equity index futures (ES, NQ): 14:30 CT = 15:30 ET
- Agricultural futures: various times
- Currency futures: 14:00 CT = 15:00 ET
- Metals: 13:30 CT = 14:30 ET
- Energy: 14:30 CT = 15:30 ET
- Bonds: 15:00 CT = 16:00 ET

Unlike crypto (funding every 8h), CME futures settle once daily.
Variation margin is credited/debited to the account based on
the difference between current settlement price and previous settlement price.

Reference:
- CME Group Settlement Procedures: https://www.cmegroup.com/clearing/operations-and-deliveries/settlement.html
- Contract Specifications: https://www.cmegroup.com/trading/equity-index/us-index/e-mini-sandp500.html
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from core_futures import (
    FuturesContractSpec,
    FuturesPosition,
    SettlementInfo,
    PositionSide,
)

logger = logging.getLogger(__name__)


# =========================
# Settlement Times by Product
# =========================


class SettlementTimeZone(str, Enum):
    """Time zone for settlement times."""

    CENTRAL = "CT"  # Chicago (CME)
    EASTERN = "ET"  # New York


# Settlement times in Eastern Time (hour, minute)
# Source: CME Group settlement procedures
SETTLEMENT_TIMES_ET: Dict[str, Tuple[int, int]] = {
    # Equity Index (14:30 CT = 15:30 ET)
    "ES": (15, 30),
    "NQ": (15, 30),
    "YM": (15, 30),
    "RTY": (15, 30),
    "MES": (15, 30),
    "MNQ": (15, 30),
    "MYM": (15, 30),
    "M2K": (15, 30),
    # Currencies (14:00 CT = 15:00 ET)
    "6E": (15, 0),
    "6J": (15, 0),
    "6B": (15, 0),
    "6A": (15, 0),
    "6C": (15, 0),
    "6S": (15, 0),
    "M6E": (15, 0),
    # Metals (13:30 CT = 14:30 ET)
    "GC": (14, 30),
    "SI": (14, 30),
    "HG": (14, 30),
    "MGC": (14, 30),
    "SIL": (14, 30),
    # Energy (14:30 CT = 15:30 ET)
    "CL": (15, 30),
    "NG": (15, 30),
    "MCL": (15, 30),
    "RB": (15, 30),
    "HO": (15, 30),
    # Bonds (15:00 CT = 16:00 ET)
    "ZB": (16, 0),
    "ZN": (16, 0),
    "ZT": (16, 0),
    "ZF": (16, 0),
    # Agricultural (various - using common 13:15 CT = 14:15 ET)
    "ZC": (14, 15),
    "ZS": (14, 15),
    "ZW": (14, 15),
    "ZM": (14, 15),
    "ZL": (14, 15),
}

# Default settlement time (equity index)
DEFAULT_SETTLEMENT_TIME_ET = (15, 30)


# =========================
# Settlement Models
# =========================


@dataclass
class VariationMarginPayment:
    """Daily variation margin payment/receipt."""

    symbol: str
    position_side: PositionSide
    position_qty: Decimal
    previous_settlement: Decimal
    current_settlement: Decimal
    price_change: Decimal
    multiplier: Decimal
    variation_margin: Decimal  # Positive = credit, Negative = debit
    settlement_date: date
    settlement_time: time
    timestamp_ms: int


@dataclass
class SettlementRecord:
    """Record of a settlement event."""

    symbol: str
    settlement_price: Decimal
    settlement_date: date
    settlement_time: time
    timestamp_ms: int
    volume: Optional[int] = None
    open_interest: Optional[int] = None


@dataclass
class DailySettlementReport:
    """Daily settlement report for an account."""

    settlement_date: date
    timestamp_ms: int
    total_variation_margin: Decimal
    payments: List[VariationMarginPayment] = field(default_factory=list)
    notes: str = ""


# =========================
# CME Settlement Engine
# =========================


class CMESettlementEngine:
    """
    CME daily settlement simulation.

    Unlike crypto (funding every 8h), CME settles once daily.
    Variation margin is credited/debited to account based on
    price movement since last settlement.

    Settlement Times (Eastern Time):
    ─────────────────────────────────────────────────────────────────
    | Product Group         | Settlement Time ET  | CT Equivalent   |
    |-----------------------|---------------------|-----------------|
    | Equity Index (ES, NQ) | 15:30 (3:30 PM)     | 14:30 (2:30 PM) |
    | Currencies (6E, 6J)   | 15:00 (3:00 PM)     | 14:00 (2:00 PM) |
    | Metals (GC, SI)       | 14:30 (2:30 PM)     | 13:30 (1:30 PM) |
    | Energy (CL, NG)       | 15:30 (3:30 PM)     | 14:30 (2:30 PM) |
    | Bonds (ZN, ZB)        | 16:00 (4:00 PM)     | 15:00 (3:00 PM) |
    ─────────────────────────────────────────────────────────────────

    Usage:
        engine = CMESettlementEngine()

        # Calculate daily variation margin
        payment = engine.calculate_variation_margin(
            position=position,
            settlement_price=Decimal("4500.00"),
            contract_spec=spec,
        )

        # Check if settlement time
        if engine.is_settlement_time(timestamp_ms, symbol="ES"):
            process_settlement()

        # Get next settlement time
        next_settlement = engine.get_next_settlement_time(now_ms, symbol="ES")
    """

    def __init__(self) -> None:
        """Initialize settlement engine."""
        self._last_settlement_prices: Dict[str, Decimal] = {}
        self._settlement_history: Dict[str, List[SettlementRecord]] = {}

    def get_settlement_time_et(self, symbol: str) -> Tuple[int, int]:
        """
        Get settlement time (hour, minute) in Eastern Time for symbol.

        Args:
            symbol: Futures symbol (ES, NQ, GC, etc.)

        Returns:
            (hour, minute) in Eastern Time
        """
        return SETTLEMENT_TIMES_ET.get(symbol.upper(), DEFAULT_SETTLEMENT_TIME_ET)

    def calculate_variation_margin(
        self,
        position: FuturesPosition,
        settlement_price: Decimal,
        contract_spec: FuturesContractSpec,
        settlement_date: Optional[date] = None,
        settlement_time: Optional[time] = None,
    ) -> VariationMarginPayment:
        """
        Calculate daily variation margin for a position.

        Variation Margin = (Settlement - Previous Settlement) × qty × multiplier

        For LONG positions:
        - Price UP → Credit (positive)
        - Price DOWN → Debit (negative)

        For SHORT positions:
        - Price DOWN → Credit (positive)
        - Price UP → Debit (negative)

        Args:
            position: Current futures position
            settlement_price: Today's settlement price
            contract_spec: Contract specification (for multiplier)
            settlement_date: Settlement date (default: today)
            settlement_time: Settlement time (default: from symbol lookup)

        Returns:
            VariationMarginPayment with calculated margin
        """
        symbol = position.symbol.upper()

        # Get previous settlement price (or use entry price if first settlement)
        previous_price = self._last_settlement_prices.get(symbol, position.entry_price)

        # Calculate price change
        price_change = settlement_price - previous_price

        # Calculate variation margin
        # quantity is always positive in FuturesPosition, side determines direction
        qty = abs(position.qty)
        multiplier = contract_spec.multiplier

        # Raw variation (assumes long)
        raw_variation = price_change * qty * multiplier

        # Adjust for position side
        if position.side == PositionSide.SHORT:
            raw_variation = -raw_variation

        # Update last settlement price
        self._last_settlement_prices[symbol] = settlement_price

        # Determine settlement time
        if settlement_date is None:
            settlement_date = date.today()

        if settlement_time is None:
            hour, minute = self.get_settlement_time_et(symbol)
            settlement_time = time(hour, minute)

        # Calculate timestamp
        settlement_dt = datetime.combine(settlement_date, settlement_time)
        timestamp_ms = int(settlement_dt.timestamp() * 1000)

        return VariationMarginPayment(
            symbol=symbol,
            position_side=position.side,
            position_qty=qty,
            previous_settlement=previous_price,
            current_settlement=settlement_price,
            price_change=price_change,
            multiplier=multiplier,
            variation_margin=raw_variation,
            settlement_date=settlement_date,
            settlement_time=settlement_time,
            timestamp_ms=timestamp_ms,
        )

    def process_daily_settlement(
        self,
        positions: List[FuturesPosition],
        settlement_prices: Dict[str, Decimal],
        contract_specs: Dict[str, FuturesContractSpec],
        settlement_date: Optional[date] = None,
    ) -> DailySettlementReport:
        """
        Process daily settlement for multiple positions.

        Args:
            positions: List of current positions
            settlement_prices: Map of symbol to settlement price
            contract_specs: Map of symbol to contract spec
            settlement_date: Settlement date (default: today)

        Returns:
            DailySettlementReport with all payments
        """
        if settlement_date is None:
            settlement_date = date.today()

        payments: List[VariationMarginPayment] = []
        total_variation = Decimal("0")

        for position in positions:
            symbol = position.symbol.upper()

            if symbol not in settlement_prices:
                logger.warning(f"No settlement price for {symbol}, skipping")
                continue

            if symbol not in contract_specs:
                logger.warning(f"No contract spec for {symbol}, skipping")
                continue

            payment = self.calculate_variation_margin(
                position=position,
                settlement_price=settlement_prices[symbol],
                contract_spec=contract_specs[symbol],
                settlement_date=settlement_date,
            )

            payments.append(payment)
            total_variation += payment.variation_margin

            # Record settlement
            self._record_settlement(
                symbol=symbol,
                settlement_price=settlement_prices[symbol],
                settlement_date=settlement_date,
                settlement_time=payment.settlement_time,
                timestamp_ms=payment.timestamp_ms,
            )

        return DailySettlementReport(
            settlement_date=settlement_date,
            timestamp_ms=int(datetime.combine(settlement_date, time(16, 0)).timestamp() * 1000),
            total_variation_margin=total_variation,
            payments=payments,
        )

    def is_settlement_time(
        self,
        timestamp_ms: int,
        symbol: Optional[str] = None,
        tolerance_minutes: int = 5,
    ) -> bool:
        """
        Check if current time is settlement time.

        Args:
            timestamp_ms: Current timestamp in milliseconds
            symbol: Optional symbol for symbol-specific time
            tolerance_minutes: Tolerance window around settlement time

        Returns:
            True if within settlement time window
        """
        dt = datetime.utcfromtimestamp(timestamp_ms / 1000)

        # Convert to Eastern Time (simplified: UTC-5)
        # In production, use proper timezone handling with pytz
        et_hour = (dt.hour - 5) % 24
        et_minute = dt.minute

        # Get expected settlement time
        if symbol:
            expected_hour, expected_minute = self.get_settlement_time_et(symbol)
        else:
            expected_hour, expected_minute = DEFAULT_SETTLEMENT_TIME_ET

        # Check if within tolerance
        current_minutes = et_hour * 60 + et_minute
        expected_minutes = expected_hour * 60 + expected_minute

        return abs(current_minutes - expected_minutes) <= tolerance_minutes

    def get_next_settlement_time(
        self,
        current_timestamp_ms: int,
        symbol: Optional[str] = None,
    ) -> int:
        """
        Get timestamp of next settlement time.

        Args:
            current_timestamp_ms: Current timestamp in milliseconds
            symbol: Optional symbol for symbol-specific time

        Returns:
            Timestamp of next settlement in milliseconds
        """
        dt = datetime.utcfromtimestamp(current_timestamp_ms / 1000)

        # Get settlement time
        if symbol:
            hour, minute = self.get_settlement_time_et(symbol)
        else:
            hour, minute = DEFAULT_SETTLEMENT_TIME_ET

        # Convert to UTC (add 5 hours to ET)
        utc_hour = (hour + 5) % 24

        # Create settlement datetime
        settlement_dt = dt.replace(hour=utc_hour, minute=minute, second=0, microsecond=0)

        # If we're past today's settlement, move to tomorrow
        if settlement_dt <= dt:
            settlement_dt += timedelta(days=1)

        # Skip weekends (CME closed)
        while settlement_dt.weekday() >= 5:  # Saturday = 5, Sunday = 6
            settlement_dt += timedelta(days=1)

        return int(settlement_dt.timestamp() * 1000)

    def get_last_settlement_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get last recorded settlement price for a symbol.

        Args:
            symbol: Futures symbol

        Returns:
            Last settlement price or None if not recorded
        """
        return self._last_settlement_prices.get(symbol.upper())

    def set_initial_settlement_price(
        self,
        symbol: str,
        price: Decimal,
    ) -> None:
        """
        Set initial settlement price for a symbol.

        Use this to initialize the settlement price when
        starting with an existing position.

        Args:
            symbol: Futures symbol
            price: Initial settlement price
        """
        self._last_settlement_prices[symbol.upper()] = price

    def get_settlement_history(
        self,
        symbol: str,
        days: int = 30,
    ) -> List[SettlementRecord]:
        """
        Get settlement history for a symbol.

        Args:
            symbol: Futures symbol
            days: Number of days of history

        Returns:
            List of SettlementRecord objects
        """
        symbol = symbol.upper()
        history = self._settlement_history.get(symbol, [])

        if not history:
            return []

        # Filter to recent days
        cutoff = date.today() - timedelta(days=days)
        return [r for r in history if r.settlement_date >= cutoff]

    def _record_settlement(
        self,
        symbol: str,
        settlement_price: Decimal,
        settlement_date: date,
        settlement_time: time,
        timestamp_ms: int,
        volume: Optional[int] = None,
        open_interest: Optional[int] = None,
    ) -> None:
        """Record a settlement event."""
        symbol = symbol.upper()

        if symbol not in self._settlement_history:
            self._settlement_history[symbol] = []

        record = SettlementRecord(
            symbol=symbol,
            settlement_price=settlement_price,
            settlement_date=settlement_date,
            settlement_time=settlement_time,
            timestamp_ms=timestamp_ms,
            volume=volume,
            open_interest=open_interest,
        )

        self._settlement_history[symbol].append(record)

        # Keep only last 365 days
        if len(self._settlement_history[symbol]) > 365:
            self._settlement_history[symbol] = self._settlement_history[symbol][-365:]

    def reset(self) -> None:
        """Reset settlement engine state."""
        self._last_settlement_prices.clear()
        self._settlement_history.clear()


# =========================
# Convenience Functions
# =========================


def create_settlement_engine() -> CMESettlementEngine:
    """Create a new CME settlement engine."""
    return CMESettlementEngine()


def calculate_variation_margin_simple(
    position_qty: Decimal,
    is_long: bool,
    previous_price: Decimal,
    settlement_price: Decimal,
    multiplier: Decimal,
) -> Decimal:
    """
    Simple variation margin calculation.

    Args:
        position_qty: Number of contracts (positive)
        is_long: True if long position
        previous_price: Previous settlement price
        settlement_price: Current settlement price
        multiplier: Contract multiplier

    Returns:
        Variation margin (positive = credit, negative = debit)
    """
    price_change = settlement_price - previous_price
    raw_variation = price_change * abs(position_qty) * multiplier

    if not is_long:
        raw_variation = -raw_variation

    return raw_variation
