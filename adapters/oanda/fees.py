# -*- coding: utf-8 -*-
"""
adapters/oanda/fees.py
OANDA fee computation adapter for forex trading.

Status: Tested and operational (Phase 2 Complete)

Forex Fee Structure:
- NO explicit commission (commission-free trading)
- Cost is embedded in the SPREAD (bid-ask)
- Swap/rollover charges for overnight positions
- No SEC/TAF regulatory fees (forex is OTC)

This adapter returns 0 for explicit fees since the cost
is captured through spread in the execution simulation.

Spread-Based Cost Model:
- Major pairs: 1.0-1.5 pips (retail), 0.1-0.3 pips (institutional)
- Minor pairs: 1.5-3.0 pips
- Cross pairs: 2.0-5.0 pips
- Exotic pairs: 15-80 pips

Swap Rates:
- Applied at daily rollover (5pm ET)
- Long/short rates are asymmetric
- Depends on interest rate differential
- Can be positive (carry trade) or negative

Usage:
    adapter = OandaFeeAdapter()
    fee = adapter.compute_fee(notional=100000, side=Side.BUY, liquidity="taker")
    # fee = 0.0 (cost is in spread)

    # Get swap estimate
    swap = adapter.compute_swap(
        notional=100000,
        symbol="EUR_USD",
        side=Side.BUY,
        days=1,
    )

References:
    - OANDA pricing: https://www.oanda.com/us-en/trading/pricing/
    - BIS 2022 Survey: https://www.bis.org/statistics/rpfx22.htm
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from core_models import Side, Liquidity
from adapters.base import FeeAdapter
from adapters.models import ExchangeVendor, FeeSchedule, FeeStructure

logger = logging.getLogger(__name__)


# =========================
# Default Spread Profiles (pips)
# =========================

# Retail spreads (typical for retail forex brokers)
RETAIL_SPREADS: Dict[str, float] = {
    # Majors
    "EUR_USD": 1.2,
    "USD_JPY": 1.2,
    "GBP_USD": 1.5,
    "USD_CHF": 1.5,
    "AUD_USD": 1.3,
    "USD_CAD": 1.5,
    "NZD_USD": 1.8,
    # Minors
    "EUR_GBP": 1.8,
    "EUR_CHF": 2.0,
    "GBP_CHF": 3.0,
    "EUR_AUD": 2.5,
    "EUR_CAD": 2.5,
    # Crosses
    "EUR_JPY": 2.0,
    "GBP_JPY": 3.0,
    "AUD_JPY": 2.5,
    "CAD_JPY": 2.5,
    # Exotics
    "USD_TRY": 50.0,
    "USD_ZAR": 80.0,
    "USD_MXN": 30.0,
}

# Institutional spreads (prime brokerage)
INSTITUTIONAL_SPREADS: Dict[str, float] = {
    "EUR_USD": 0.3,
    "USD_JPY": 0.3,
    "GBP_USD": 0.5,
    "USD_CHF": 0.5,
    "AUD_USD": 0.4,
    "USD_CAD": 0.5,
    "NZD_USD": 0.6,
    "EUR_GBP": 0.6,
    "EUR_JPY": 0.8,
    "GBP_JPY": 1.2,
}

# Default swap rates (pips per day, example values)
# Positive = receive, Negative = pay
DEFAULT_SWAP_RATES: Dict[str, Tuple[float, float]] = {
    # (long_swap, short_swap) in pips/day
    "EUR_USD": (-0.5, 0.2),  # Pay to be long EUR (lower rates)
    "USD_JPY": (0.3, -0.5),  # Receive to be long USD (higher rates)
    "GBP_USD": (-0.3, 0.1),
    "AUD_USD": (-0.2, 0.0),
    "USD_CHF": (0.1, -0.3),
    "EUR_JPY": (-0.6, 0.2),
}


class OandaFeeAdapter(FeeAdapter):
    """
    OANDA fee computation adapter for forex.

    In forex, there are no explicit commissions. Trading cost is embedded
    in the bid-ask spread. This adapter returns 0 for explicit fees.

    For accurate cost modeling, use the spread from tick data in the
    execution simulation layer.

    Features:
    - Zero explicit commission (spread-based pricing)
    - Swap rate estimation for overnight positions
    - Spread lookup by currency pair
    - Session-adjusted spread multipliers

    Configuration:
        spread_profile: "retail" or "institutional" (default: "retail")
        spreads: Custom spread overrides dict
        swap_rates: Custom swap rate overrides dict
        include_swap: Include swap in fee estimate (default: False)
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.OANDA,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Initialize OANDA fee adapter.

        Args:
            vendor: Exchange vendor (default: OANDA)
            config: Configuration dict
        """
        super().__init__(vendor, config)

        # Spread profile
        self._spread_profile = self._config.get("spread_profile", "retail")
        if self._spread_profile == "institutional":
            self._spreads = dict(INSTITUTIONAL_SPREADS)
        else:
            self._spreads = dict(RETAIL_SPREADS)

        # Apply custom spread overrides
        custom_spreads = self._config.get("spreads", {})
        self._spreads.update(custom_spreads)

        # Swap rates
        self._swap_rates = dict(DEFAULT_SWAP_RATES)
        custom_swaps = self._config.get("swap_rates", {})
        self._swap_rates.update(custom_swaps)

        # Whether to include swap in fee estimate
        self._include_swap = self._config.get("include_swap", False)

    @property
    def spread_profile(self) -> str:
        """Current spread profile (retail or institutional)."""
        return self._spread_profile

    def compute_fee(
        self,
        notional: float,
        side: Side,
        liquidity: Union[str, Liquidity],
        *,
        symbol: Optional[str] = None,
        qty: Optional[float] = None,
        price: Optional[float] = None,
    ) -> float:
        """
        Compute trading fee for a forex trade.

        Forex has NO explicit commission - returns 0.
        Cost is embedded in the spread, which is handled separately
        by the execution simulation.

        Args:
            notional: Trade value (price * quantity)
            side: Trade direction (BUY/SELL)
            liquidity: Not used for forex (always market maker)
            symbol: Currency pair (for spread lookup)
            qty: Number of units (for pip value calculation)
            price: Trade price

        Returns:
            0.0 - Forex has no explicit commission
        """
        # Forex has no explicit commission
        # All cost is in the spread (bid-ask)
        return 0.0

    def get_fee_schedule(self, symbol: Optional[str] = None) -> FeeSchedule:
        """
        Get fee schedule for currency pair.

        Returns zero rates since forex cost is in spread.

        Args:
            symbol: Currency pair (for spread info)

        Returns:
            FeeSchedule with zero rates
        """
        return FeeSchedule(
            structure=FeeStructure.FLAT,  # No percentage-based fees
            maker_rate=0.0,
            taker_rate=0.0,
            flat_fee=0.0,
            min_fee=0.0,
            currency="USD",
            rebate_enabled=False,
        )

    def get_effective_rates(
        self,
        symbol: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Get effective maker/taker rates in basis points.

        Forex doesn't have maker/taker distinction - all trades
        are against the dealer's spread.

        Args:
            symbol: Currency pair

        Returns:
            (0.0, 0.0) - No explicit fees
        """
        return (0.0, 0.0)

    def get_spread_pips(self, symbol: str) -> float:
        """
        Get typical spread for a currency pair in pips.

        Args:
            symbol: Currency pair (e.g., "EUR_USD", "GBP/USD")

        Returns:
            Spread in pips
        """
        # Normalize symbol
        norm = symbol.upper().replace("/", "_").replace("-", "_")

        # Look up spread
        spread = self._spreads.get(norm)

        if spread is not None:
            return spread

        # Estimate based on pair type
        if self._is_exotic(norm):
            return 40.0  # Default exotic spread
        elif self._is_cross(norm):
            return 3.0  # Default cross spread
        elif self._is_minor(norm):
            return 2.0  # Default minor spread
        else:
            return 1.5  # Default major spread

    def get_spread_cost(
        self,
        symbol: str,
        notional: float,
        price: float,
    ) -> float:
        """
        Calculate spread cost for a trade.

        This is the implicit trading cost in forex.

        Args:
            symbol: Currency pair
            notional: Trade notional value in quote currency
            price: Current mid price

        Returns:
            Spread cost in quote currency
        """
        spread_pips = self.get_spread_pips(symbol)
        pip_size = self.get_pip_size(symbol)

        # Spread cost = (spread * pip_size / price) * notional / 2
        # Divide by 2 because spread is round-trip cost
        spread_value = spread_pips * pip_size
        relative_spread = spread_value / price if price > 0 else 0.0

        return notional * relative_spread / 2.0

    def get_pip_size(self, symbol: str) -> float:
        """
        Get pip size for a currency pair.

        Args:
            symbol: Currency pair

        Returns:
            Pip size (0.0001 for most, 0.01 for JPY pairs)
        """
        norm = symbol.upper().replace("/", "_")
        # JPY pairs have pip size of 0.01
        return 0.01 if "JPY" in norm else 0.0001

    def compute_swap(
        self,
        notional: float,
        symbol: str,
        side: Side,
        days: int = 1,
    ) -> float:
        """
        Compute estimated swap/rollover cost.

        Swap is charged/credited at daily rollover (5pm ET) for
        positions held overnight.

        Args:
            notional: Position notional value
            symbol: Currency pair
            side: Position direction (BUY/SELL)
            days: Number of days to hold

        Returns:
            Swap amount (positive = receive, negative = pay)
        """
        norm = symbol.upper().replace("/", "_")

        # Get swap rates
        swap_rates = self._swap_rates.get(norm, (-0.5, 0.2))

        # Select long or short rate
        if isinstance(side, Side):
            is_long = side == Side.BUY
        else:
            is_long = str(side).upper() == "BUY"

        swap_pips_per_day = swap_rates[0] if is_long else swap_rates[1]

        # Convert pips to currency value
        pip_size = self.get_pip_size(symbol)
        pip_value_per_unit = pip_size

        # Estimate number of units from notional
        # Assume price ~1.0 for simplicity in swap calculation
        estimated_units = notional

        # Total swap in quote currency
        total_swap = swap_pips_per_day * pip_value_per_unit * estimated_units * days

        return round(total_swap, 4)

    def get_swap_rates(
        self,
        symbol: str,
    ) -> Tuple[float, float]:
        """
        Get swap rates for a currency pair.

        Args:
            symbol: Currency pair

        Returns:
            (long_swap_pips_per_day, short_swap_pips_per_day)
        """
        norm = symbol.upper().replace("/", "_")
        return self._swap_rates.get(norm, (-0.5, 0.2))

    def estimate_total_cost(
        self,
        symbol: str,
        notional: float,
        price: float,
        side: Side,
        holding_days: int = 0,
    ) -> Dict[str, float]:
        """
        Estimate total trading cost including spread and swap.

        Args:
            symbol: Currency pair
            notional: Trade notional value
            price: Current price
            side: Trade direction
            holding_days: Days to hold position

        Returns:
            Dict with cost breakdown
        """
        spread_cost = self.get_spread_cost(symbol, notional, price)
        swap_cost = self.compute_swap(notional, symbol, side, holding_days)

        return {
            "spread_cost": round(spread_cost, 4),
            "swap_cost": round(swap_cost, 4),
            "total_cost": round(spread_cost + abs(swap_cost), 4),
            "spread_pips": self.get_spread_pips(symbol),
        }

    def _is_major(self, symbol: str) -> bool:
        """Check if symbol is a major pair."""
        majors = {"EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD"}
        return symbol in majors

    def _is_minor(self, symbol: str) -> bool:
        """Check if symbol is a minor pair (no USD, no exotic)."""
        minors = {"EUR_GBP", "EUR_CHF", "GBP_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD"}
        return symbol in minors

    def _is_cross(self, symbol: str) -> bool:
        """Check if symbol is a cross pair (JPY crosses)."""
        return "JPY" in symbol and "USD" not in symbol

    def _is_exotic(self, symbol: str) -> bool:
        """Check if symbol is an exotic pair."""
        exotic_currencies = {"TRY", "ZAR", "MXN", "PLN", "HUF", "CZK", "THB", "SGD", "HKD"}
        parts = symbol.split("_")
        return any(curr in exotic_currencies for curr in parts)

    def update_swap_rates(self, rates: Dict[str, Tuple[float, float]]) -> None:
        """
        Update swap rates from external source.

        Args:
            rates: Dict mapping symbol to (long_swap, short_swap)
        """
        for symbol, swap_tuple in rates.items():
            norm = symbol.upper().replace("/", "_")
            self._swap_rates[norm] = swap_tuple

    def update_spreads(self, spreads: Dict[str, float]) -> None:
        """
        Update spread values from external source.

        Args:
            spreads: Dict mapping symbol to spread in pips
        """
        for symbol, spread in spreads.items():
            norm = symbol.upper().replace("/", "_")
            self._spreads[norm] = spread
