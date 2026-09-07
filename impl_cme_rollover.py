# -*- coding: utf-8 -*-
"""
impl_cme_rollover.py
Contract rollover management for CME Group futures.

Handles:
- Contract expiration tracking
- Roll date calculation
- Continuous contract price adjustment
- Roll spread tracking

Standard Roll Dates:
- Equity Index (ES, NQ): 8 business days before expiry (2nd Thursday before 3rd Friday)
- Metals (GC, SI): Last trading day of month before delivery month
- Energy (CL, NG): ~3 trading days before expiry
- Currencies (6E, 6J): 2nd business day before 3rd Wednesday

Contract Month Codes:
F=Jan, G=Feb, H=Mar, J=Apr, K=May, M=Jun
N=Jul, Q=Aug, U=Sep, V=Oct, X=Nov, Z=Dec

Reference:
- CME Expiration Calendar: https://www.cmegroup.com/tools-information/holiday-calendar.html
- Contract Specifications: https://www.cmegroup.com/trading/products/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from core_futures import (
    FuturesContractSpec,
    ContractRollover,
    ContractType,
)

logger = logging.getLogger(__name__)


# =========================
# Contract Month Codes
# =========================

MONTH_CODES = {
    1: "F",  # January
    2: "G",  # February
    3: "H",  # March
    4: "J",  # April
    5: "K",  # May
    6: "M",  # June
    7: "N",  # July
    8: "Q",  # August
    9: "U",  # September
    10: "V",  # October
    11: "X",  # November
    12: "Z",  # December
}

MONTH_CODE_TO_NUM = {v: k for k, v in MONTH_CODES.items()}


# =========================
# Contract Cycles
# =========================


class ContractCycle(str, Enum):
    """Standard contract listing cycles."""

    # Quarterly (Mar, Jun, Sep, Dec)
    QUARTERLY = "quarterly"  # H, M, U, Z
    # Monthly
    MONTHLY = "monthly"  # All months
    # Bi-monthly
    BI_MONTHLY = "bi_monthly"  # Every 2 months


# Contract cycles by product
PRODUCT_CYCLES: Dict[str, ContractCycle] = {
    # Equity Index - Quarterly
    "ES": ContractCycle.QUARTERLY,
    "NQ": ContractCycle.QUARTERLY,
    "YM": ContractCycle.QUARTERLY,
    "RTY": ContractCycle.QUARTERLY,
    "MES": ContractCycle.QUARTERLY,
    "MNQ": ContractCycle.QUARTERLY,
    # Currencies - Quarterly
    "6E": ContractCycle.QUARTERLY,
    "6J": ContractCycle.QUARTERLY,
    "6B": ContractCycle.QUARTERLY,
    # Metals - Bi-Monthly (but most liquid are Feb, Apr, Jun, Aug, Oct, Dec)
    "GC": ContractCycle.BI_MONTHLY,
    "SI": ContractCycle.BI_MONTHLY,
    # Energy - Monthly
    "CL": ContractCycle.MONTHLY,
    "NG": ContractCycle.MONTHLY,
    # Bonds - Quarterly
    "ZN": ContractCycle.QUARTERLY,
    "ZB": ContractCycle.QUARTERLY,
    # Grains - varies but treated as bi-monthly for simplicity
    "ZC": ContractCycle.BI_MONTHLY,
    "ZS": ContractCycle.BI_MONTHLY,
    "ZW": ContractCycle.BI_MONTHLY,
}

# Quarterly months
QUARTERLY_MONTHS = {3, 6, 9, 12}  # H, M, U, Z

# Roll days before expiry by product type
ROLL_DAYS_BEFORE: Dict[str, int] = {
    # Equity Index: 8 business days before expiry
    "ES": 8,
    "NQ": 8,
    "YM": 8,
    "RTY": 8,
    "MES": 8,
    "MNQ": 8,
    "MYM": 8,
    "M2K": 8,
    # Currencies: 2 business days before expiry
    "6E": 2,
    "6J": 2,
    "6B": 2,
    "6A": 2,
    "6C": 2,
    # Metals: 3 business days before last trading day
    "GC": 3,
    "SI": 3,
    "HG": 3,
    # Energy: 3 business days before expiry
    "CL": 3,
    "NG": 3,
    # Bonds: 7 business days before first delivery day
    "ZN": 7,
    "ZB": 7,
    "ZT": 7,
    "ZF": 7,
}

DEFAULT_ROLL_DAYS = 8


# =========================
# Rollover Models
# =========================


@dataclass
class ContractInfo:
    """Information about a specific contract."""

    symbol: str
    base_symbol: str
    month_code: str
    year: int
    expiry_date: date
    first_notice_date: Optional[date] = None
    last_trading_date: Optional[date] = None
    is_front_month: bool = False


@dataclass
class RolloverInfo:
    """Information about a rollover event."""

    from_contract: ContractInfo
    to_contract: ContractInfo
    roll_date: date
    roll_spread: Optional[Decimal] = None  # Price difference (to - from)
    adjustment_factor: Optional[Decimal] = None  # For continuous contracts
    volume_front: Optional[int] = None
    volume_back: Optional[int] = None


@dataclass
class ContinuousContractAdjustment:
    """Adjustment for continuous contract series."""

    symbol: str
    roll_date: date
    front_price: Decimal
    back_price: Decimal
    adjustment: Decimal  # Amount to add/subtract for backward adjustment
    cumulative_adjustment: Decimal  # Total adjustment from current


# =========================
# Contract Rollover Manager
# =========================


class ContractRolloverManager:
    """
    Contract rollover management for CME Group futures.

    Handles:
    - Contract expiration tracking
    - Roll date calculation
    - Continuous contract price adjustment
    - Roll spread tracking

    Standard Roll Timing (business days before expiry):
    ─────────────────────────────────────────────────────────────────
    | Product Group         | Roll Days Before | Notes              |
    |-----------------------|------------------|--------------------|
    | Equity Index (ES, NQ) | 8                | 2nd Thu before 3rd Fri |
    | Currencies (6E, 6J)   | 2                | Before 3rd Wed     |
    | Metals (GC, SI)       | 3                | Before last trading day |
    | Energy (CL, NG)       | 3                | Before expiry      |
    | Bonds (ZN, ZB)        | 7                | Before 1st delivery |
    ─────────────────────────────────────────────────────────────────

    Usage:
        manager = ContractRolloverManager()

        # Check if should roll
        if manager.should_roll("ES", date.today()):
            roll_info = manager.get_roll_info("ES", date.today())
            # Execute roll...

        # Get front month
        front = manager.get_front_month("ES", date.today())
        print(f"Front month: {front.symbol}")

        # Build continuous series
        adj_prices = manager.build_continuous_series(
            "ES",
            price_history,
            roll_history
        )
    """

    def __init__(
        self,
        expiration_calendar: Optional[Dict[str, List[date]]] = None,
    ) -> None:
        """
        Initialize rollover manager.

        Args:
            expiration_calendar: Map of base symbol to list of expiration dates.
                                If None, will calculate expiry dates using rules.
        """
        self._calendar = expiration_calendar or {}
        self._roll_history: Dict[str, List[RolloverInfo]] = {}
        self._adjustments: Dict[str, List[ContinuousContractAdjustment]] = {}

    def get_roll_days(self, symbol: str) -> int:
        """Get number of business days before expiry to roll."""
        return ROLL_DAYS_BEFORE.get(symbol.upper(), DEFAULT_ROLL_DAYS)

    def should_roll(
        self,
        symbol: str,
        current_date: date,
        include_weekend_buffer: bool = True,
    ) -> bool:
        """
        Check if contract should be rolled.

        Args:
            symbol: Base futures symbol (ES, NQ, etc.)
            current_date: Current date
            include_weekend_buffer: If True, roll earlier if expiry is after weekend

        Returns:
            True if should roll to next contract
        """
        symbol = symbol.upper()
        expiry = self._get_current_expiry(symbol, current_date)

        if expiry is None:
            return False

        # Calculate roll date
        roll_days = self.get_roll_days(symbol)
        roll_date = self._subtract_business_days(expiry, roll_days)

        # Add weekend buffer if needed
        if include_weekend_buffer:
            # If roll date is Monday, roll on previous Friday
            if roll_date.weekday() == 0:  # Monday
                roll_date = roll_date - timedelta(days=3)

        return current_date >= roll_date

    def get_roll_date(
        self,
        symbol: str,
        current_date: date,
    ) -> Optional[date]:
        """
        Get the roll date for current front month contract.

        Args:
            symbol: Base futures symbol
            current_date: Current date

        Returns:
            Roll date or None if no expiry found
        """
        symbol = symbol.upper()
        expiry = self._get_current_expiry(symbol, current_date)

        if expiry is None:
            return None

        roll_days = self.get_roll_days(symbol)
        return self._subtract_business_days(expiry, roll_days)

    def get_roll_info(
        self,
        symbol: str,
        current_date: date,
        front_price: Optional[Decimal] = None,
        back_price: Optional[Decimal] = None,
    ) -> Optional[RolloverInfo]:
        """
        Get rollover information for current contract.

        Args:
            symbol: Base futures symbol
            current_date: Current date
            front_price: Current front month price
            back_price: Current back month price

        Returns:
            RolloverInfo or None if not in roll period
        """
        symbol = symbol.upper()

        if not self.should_roll(symbol, current_date):
            return None

        front_contract = self.get_front_month(symbol, current_date)
        back_contract = self.get_next_month(symbol, current_date)

        if front_contract is None or back_contract is None:
            return None

        roll_spread = None
        adjustment_factor = None

        if front_price is not None and back_price is not None:
            roll_spread = back_price - front_price
            # For backward adjustment: add this to historical prices
            adjustment_factor = roll_spread

        roll_date = self.get_roll_date(symbol, current_date)

        return RolloverInfo(
            from_contract=front_contract,
            to_contract=back_contract,
            roll_date=roll_date or current_date,
            roll_spread=roll_spread,
            adjustment_factor=adjustment_factor,
        )

    def get_front_month(
        self,
        symbol: str,
        current_date: date,
    ) -> Optional[ContractInfo]:
        """
        Get front month contract info.

        Args:
            symbol: Base futures symbol
            current_date: Current date

        Returns:
            ContractInfo for front month or None
        """
        symbol = symbol.upper()
        expiry = self._get_current_expiry(symbol, current_date)

        if expiry is None:
            # Calculate from rules
            expiry = self._calculate_next_expiry(symbol, current_date)

        if expiry is None:
            return None

        month_code = MONTH_CODES[expiry.month]
        year = expiry.year

        return ContractInfo(
            symbol=f"{symbol}{month_code}{str(year)[2:]}",
            base_symbol=symbol,
            month_code=month_code,
            year=year,
            expiry_date=expiry,
            is_front_month=True,
        )

    def get_next_month(
        self,
        symbol: str,
        current_date: date,
    ) -> Optional[ContractInfo]:
        """
        Get next month (back month) contract info.

        Args:
            symbol: Base futures symbol
            current_date: Current date

        Returns:
            ContractInfo for next month or None
        """
        symbol = symbol.upper()
        front = self.get_front_month(symbol, current_date)

        if front is None:
            return None

        # Get next expiry after front month
        next_date = front.expiry_date + timedelta(days=1)
        next_expiry = self._get_current_expiry(symbol, next_date)

        if next_expiry is None:
            next_expiry = self._calculate_next_expiry(symbol, next_date)

        if next_expiry is None:
            return None

        month_code = MONTH_CODES[next_expiry.month]
        year = next_expiry.year

        return ContractInfo(
            symbol=f"{symbol}{month_code}{str(year)[2:]}",
            base_symbol=symbol,
            month_code=month_code,
            year=year,
            expiry_date=next_expiry,
            is_front_month=False,
        )

    def record_roll(
        self,
        roll_info: RolloverInfo,
    ) -> None:
        """
        Record a rollover event.

        Args:
            roll_info: Rollover information
        """
        symbol = roll_info.from_contract.base_symbol

        if symbol not in self._roll_history:
            self._roll_history[symbol] = []

        self._roll_history[symbol].append(roll_info)

        # Record adjustment if present
        if roll_info.roll_spread is not None:
            if symbol not in self._adjustments:
                self._adjustments[symbol] = []

            # Calculate cumulative adjustment
            prev_cumulative = Decimal("0")
            if self._adjustments[symbol]:
                prev_cumulative = self._adjustments[symbol][-1].cumulative_adjustment

            self._adjustments[symbol].append(
                ContinuousContractAdjustment(
                    symbol=symbol,
                    roll_date=roll_info.roll_date,
                    front_price=Decimal("0"),  # Would need actual prices
                    back_price=Decimal("0"),
                    adjustment=roll_info.roll_spread,
                    cumulative_adjustment=prev_cumulative + roll_info.roll_spread,
                )
            )

    def get_roll_history(
        self,
        symbol: str,
        lookback_days: int = 365,
    ) -> List[RolloverInfo]:
        """
        Get rollover history for a symbol.

        Args:
            symbol: Base futures symbol
            lookback_days: Number of days to look back

        Returns:
            List of RolloverInfo objects
        """
        symbol = symbol.upper()
        history = self._roll_history.get(symbol, [])

        if not history:
            return []

        cutoff = date.today() - timedelta(days=lookback_days)
        return [r for r in history if r.roll_date >= cutoff]

    def adjust_price_for_continuous(
        self,
        symbol: str,
        price: Decimal,
        price_date: date,
        adjustment_method: str = "backward",
    ) -> Decimal:
        """
        Adjust price for continuous contract series.

        Args:
            symbol: Base futures symbol
            price: Raw price
            price_date: Date of price
            adjustment_method: "backward" (add to historical) or "forward" (subtract from current)

        Returns:
            Adjusted price
        """
        symbol = symbol.upper()
        adjustments = self._adjustments.get(symbol, [])

        if not adjustments:
            return price

        # Find cumulative adjustment at price_date
        cumulative = Decimal("0")
        for adj in adjustments:
            if adj.roll_date > price_date:
                break
            cumulative = adj.cumulative_adjustment

        if adjustment_method == "backward":
            # Add adjustment to historical prices
            return price + cumulative
        else:
            # Subtract from current
            return price - cumulative

    def get_contract_symbol(
        self,
        base_symbol: str,
        month: int,
        year: int,
    ) -> str:
        """
        Build contract symbol from components.

        Args:
            base_symbol: Base symbol (ES, NQ, etc.)
            month: Contract month (1-12)
            year: Contract year (2024, 2025, etc.)

        Returns:
            Full contract symbol (e.g., ESH24)
        """
        month_code = MONTH_CODES[month]
        year_suffix = str(year)[2:]  # Last 2 digits
        return f"{base_symbol.upper()}{month_code}{year_suffix}"

    def parse_contract_symbol(self, symbol: str) -> Optional[ContractInfo]:
        """
        Parse contract symbol into components.

        Args:
            symbol: Full contract symbol (e.g., ESH24)

        Returns:
            ContractInfo or None if invalid
        """
        if len(symbol) < 3:
            return None

        try:
            # Find month code position
            for i in range(len(symbol) - 2, 0, -1):
                if symbol[i] in MONTH_CODE_TO_NUM:
                    base_symbol = symbol[:i]
                    month_code = symbol[i]
                    year_str = symbol[i + 1 :]

                    month = MONTH_CODE_TO_NUM[month_code]
                    year = 2000 + int(year_str) if len(year_str) == 2 else int(year_str)

                    # Calculate approximate expiry
                    expiry = self._calculate_expiry_for_month(base_symbol, month, year)

                    return ContractInfo(
                        symbol=symbol,
                        base_symbol=base_symbol,
                        month_code=month_code,
                        year=year,
                        expiry_date=expiry,
                    )
        except (ValueError, KeyError):
            pass

        return None

    # =========================
    # Private Helper Methods
    # =========================

    def _get_current_expiry(
        self,
        symbol: str,
        current_date: date,
    ) -> Optional[date]:
        """Get current front month expiry from calendar."""
        expirations = self._calendar.get(symbol, [])
        for exp in sorted(expirations):
            if exp > current_date:
                return exp
        return None

    def _calculate_next_expiry(
        self,
        symbol: str,
        current_date: date,
    ) -> Optional[date]:
        """Calculate next expiry date using rules."""
        symbol = symbol.upper()
        cycle = PRODUCT_CYCLES.get(symbol, ContractCycle.QUARTERLY)

        # Find next valid contract month
        month = current_date.month
        year = current_date.year

        for _ in range(24):  # Max 2 years lookahead
            month += 1
            if month > 12:
                month = 1
                year += 1

            if cycle == ContractCycle.QUARTERLY:
                if month not in QUARTERLY_MONTHS:
                    continue
            elif cycle == ContractCycle.BI_MONTHLY:
                if month % 2 != 0:  # Even months only
                    continue
            # MONTHLY accepts all months

            expiry = self._calculate_expiry_for_month(symbol, month, year)
            if expiry > current_date:
                return expiry

        return None

    def _calculate_expiry_for_month(
        self,
        symbol: str,
        month: int,
        year: int,
    ) -> date:
        """Calculate expiry date for a specific contract month."""
        # Default: 3rd Friday of the month
        # Find first day of month
        first_day = date(year, month, 1)

        # Find first Friday
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)

        # 3rd Friday
        third_friday = first_friday + timedelta(weeks=2)

        return third_friday

    def _subtract_business_days(
        self,
        from_date: date,
        days: int,
    ) -> date:
        """Subtract business days from a date."""
        result = from_date
        remaining = days

        while remaining > 0:
            result -= timedelta(days=1)
            if result.weekday() < 5:  # Monday-Friday
                remaining -= 1

        return result

    def set_expiration_calendar(
        self,
        symbol: str,
        expirations: List[date],
    ) -> None:
        """
        Set expiration calendar for a symbol.

        Args:
            symbol: Base futures symbol
            expirations: List of expiration dates
        """
        self._calendar[symbol.upper()] = sorted(expirations)


# =========================
# Convenience Functions
# =========================


def create_rollover_manager(
    expiration_calendar: Optional[Dict[str, List[date]]] = None,
) -> ContractRolloverManager:
    """Create a new rollover manager."""
    return ContractRolloverManager(expiration_calendar)


def get_contract_month_code(month: int) -> str:
    """Get month code for a month number (1-12)."""
    return MONTH_CODES.get(month, "?")


def get_month_from_code(code: str) -> int:
    """Get month number from code (F, G, H, etc.)."""
    return MONTH_CODE_TO_NUM.get(code.upper(), 0)
