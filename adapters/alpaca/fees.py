# -*- coding: utf-8 -*-
"""
adapters/alpaca/fees.py
Alpaca fee computation adapter.

Alpaca offers commission-free stock trading, with some fees for:
- Options trading (per-contract fees)
- Regulatory fees (TAF, SEC fees) - typically passed through
- Margin interest
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Tuple, Union

from core_models import Side, Liquidity
from adapters.base import FeeAdapter
from adapters.models import ExchangeVendor, FeeSchedule, FeeStructure

logger = logging.getLogger(__name__)


class AlpacaFeeAdapter(FeeAdapter):
    """
    Alpaca fee computation adapter.

    Alpaca provides commission-free stock trading.
    This adapter returns zero fees for most trades.

    Fees that may apply:
    - SEC fee: ~$0.0000278 per dollar of sale proceeds
    - TAF fee: $0.000166 per share sold (max $8.30)
    - Options: ~$0.50-1.00 per contract (if enabled)

    Configuration:
        include_regulatory_fees: Include SEC/TAF fees (default: True)
        options_per_contract_fee: Fee per options contract (default: 0.65)
    """

    # Regulatory fee rates (as of 2024, subject to change)
    SEC_FEE_RATE = 0.0000278  # Per dollar of sale proceeds
    TAF_FEE_RATE = 0.000166  # Per share sold
    TAF_MAX_FEE = 8.30  # Maximum TAF fee per trade

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.ALPACA,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor, config)

        self._include_regulatory = self._config.get("include_regulatory_fees", True)
        self._options_fee = float(self._config.get("options_per_contract_fee", 0.65))

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
        Compute trading fee for a stock trade.

        Alpaca is commission-free, but regulatory fees may apply to sales.

        Args:
            notional: Trade value (price * quantity)
            side: Trade direction (BUY/SELL)
            liquidity: "maker" or "taker" (not used for Alpaca)
            symbol: Stock symbol
            qty: Number of shares
            price: Trade price

        Returns:
            Fee amount in USD (usually 0, or small regulatory fees on sales)
        """
        # Commission-free for buys
        if isinstance(side, Side):
            is_sell = side == Side.SELL
        else:
            is_sell = str(side).upper() == "SELL"

        if not is_sell:
            return 0.0

        # For sells, calculate regulatory fees if enabled
        if not self._include_regulatory:
            return 0.0

        fee = 0.0

        # SEC fee (on sale proceeds)
        sec_fee = abs(notional) * self.SEC_FEE_RATE
        fee += sec_fee

        # TAF fee (per share sold)
        if qty is not None:
            taf_fee = min(abs(qty) * self.TAF_FEE_RATE, self.TAF_MAX_FEE)
            fee += taf_fee

        return round(fee, 4)  # Round to 4 decimal places

    def get_fee_schedule(self, symbol: Optional[str] = None) -> FeeSchedule:
        """
        Get fee schedule.

        Alpaca uses commission-free trading with optional regulatory fees.

        Args:
            symbol: Stock symbol (not used)

        Returns:
            FeeSchedule with zero rates
        """
        return FeeSchedule(
            structure=FeeStructure.FLAT,
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
        Get effective maker/taker rates.

        Alpaca doesn't distinguish maker/taker - all trades are commission-free.

        Args:
            symbol: Stock symbol (not used)

        Returns:
            (0.0, 0.0) - Zero fees
        """
        return (0.0, 0.0)

    def compute_options_fee(
        self,
        contracts: int,
        opening: bool = True,
    ) -> float:
        """
        Compute options trading fee.

        Args:
            contracts: Number of option contracts
            opening: True for opening, False for closing

        Returns:
            Fee amount in USD
        """
        base_fee = abs(contracts) * self._options_fee

        # Regulatory fees on options
        # SEC/TAF fees are typically lower for options
        regulatory = 0.0
        if self._include_regulatory and not opening:
            # Small regulatory fee on closing
            regulatory = abs(contracts) * 0.02  # Approximate

        return round(base_fee + regulatory, 2)

    def estimate_regulatory_fees(
        self,
        notional: float,
        qty: float,
    ) -> dict:
        """
        Estimate regulatory fees breakdown.

        Args:
            notional: Trade value
            qty: Number of shares

        Returns:
            Dict with fee breakdown
        """
        sec_fee = abs(notional) * self.SEC_FEE_RATE
        taf_fee = min(abs(qty) * self.TAF_FEE_RATE, self.TAF_MAX_FEE)

        return {
            "sec_fee": round(sec_fee, 4),
            "taf_fee": round(taf_fee, 4),
            "total": round(sec_fee + taf_fee, 4),
        }
