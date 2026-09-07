# -*- coding: utf-8 -*-
"""
adapters/binance/fees.py
Binance fee computation adapter.

Wraps existing fees.py and impl_fees.py functionality into the FeeAdapter interface.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from core_models import Side, Liquidity
from adapters.base import FeeAdapter
from adapters.models import ExchangeVendor, FeeSchedule, FeeStructure

logger = logging.getLogger(__name__)


class BinanceFeeAdapter(FeeAdapter):
    """
    Binance fee computation adapter.

    Wraps FeesModel from fees.py to implement the FeeAdapter interface.

    Configuration:
        maker_bps: Default maker fee in basis points (default: 10)
        taker_bps: Default taker fee in basis points (default: 10)
        use_bnb_discount: Whether to apply BNB discount (default: False)
        vip_tier: VIP tier level (default: 0)
        fee_cache_path: Path to cached fee data
        api_key: For fetching account-specific fees
        api_secret: For fetching account-specific fees
    """

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.BINANCE,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor, config)

        # Lazy-loaded fee model
        self._fee_model = None

    def _get_fee_model(self):
        """Lazy initialization of FeesModel."""
        if self._fee_model is None:
            from fees import FeesModel

            self._fee_model = FeesModel(
                maker_bps=float(self._config.get("maker_bps", 10.0)),
                taker_bps=float(self._config.get("taker_bps", 10.0)),
                use_bnb_discount=bool(self._config.get("use_bnb_discount", False)),
                vip_tier=int(self._config.get("vip_tier", 0)),
            )

            # Try to load cached fees if available
            fee_cache_path = self._config.get("fee_cache_path")
            if fee_cache_path:
                try:
                    self._load_fee_cache(fee_cache_path)
                except Exception as e:
                    logger.warning(f"Failed to load fee cache: {e}")

        return self._fee_model

    def _load_fee_cache(self, path: str) -> None:
        """Load fee data from cache file."""
        import json
        from pathlib import Path

        cache_path = Path(path)
        if not cache_path.exists():
            return

        with open(cache_path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            from fees import FeesModel

            self._fee_model = FeesModel.from_dict(data)

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
        Compute trading fee for a trade.

        Args:
            notional: Trade value (price * quantity)
            side: Trade direction (BUY/SELL)
            liquidity: "maker" or "taker"
            symbol: Trading symbol for symbol-specific fees
            qty: Trade quantity (not used for percentage fees)
            price: Trade price (not used for percentage fees)

        Returns:
            Fee amount in quote currency
        """
        model = self._get_fee_model()

        # Normalize liquidity to string
        if isinstance(liquidity, Liquidity):
            liq_str = liquidity.value.lower()
        else:
            liq_str = str(liquidity).lower()

        # Compute fee using FeesModel
        fee = model.compute(
            side=side.value if isinstance(side, Side) else str(side),
            price=price or 1.0,  # We use notional directly
            qty=notional / (price or 1.0) if price else qty or notional,
            liquidity=liq_str,
            symbol=symbol,
        )

        return float(fee)

    def get_fee_schedule(self, symbol: Optional[str] = None) -> FeeSchedule:
        """
        Get fee schedule for symbol.

        Args:
            symbol: Trading symbol (optional)

        Returns:
            FeeSchedule with fee rates
        """
        model = self._get_fee_model()

        maker_bps = model.get_fee_bps(symbol, is_maker=True)
        taker_bps = model.get_fee_bps(symbol, is_maker=False)

        return FeeSchedule(
            structure=FeeStructure.PERCENTAGE,
            maker_rate=maker_bps,
            taker_rate=taker_bps,
            currency="USDT",  # Most common for Binance
            rebate_enabled=False,
        )

    def get_effective_rates(
        self,
        symbol: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Get effective maker/taker rates in basis points.

        Args:
            symbol: Trading symbol (optional)

        Returns:
            (maker_bps, taker_bps)
        """
        model = self._get_fee_model()

        # Apply discount multipliers
        maker_bps = model.get_fee_bps(symbol, is_maker=True)
        taker_bps = model.get_fee_bps(symbol, is_maker=False)

        # Apply discounts
        maker_mult = model._discount_multiplier(symbol, is_maker=True)
        taker_mult = model._discount_multiplier(symbol, is_maker=False)

        return (maker_bps * maker_mult, taker_bps * taker_mult)

    def fetch_account_fees(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetch account-specific fee rates from Binance.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret

        Returns:
            Dict with fee information
        """
        key = api_key or self._config.get("api_key")
        secret = api_secret or self._config.get("api_secret")

        if not key or not secret:
            raise ValueError("API key and secret required for account fee fetch")

        from adapters.binance_spot_private import fetch_account_fee_info

        info = fetch_account_fee_info(api_key=key, api_secret=secret)

        # Update internal model with account fees
        if self._fee_model is not None:
            if info.vip_tier is not None:
                self._fee_model.vip_tier = info.vip_tier
            if info.maker_bps is not None:
                self._fee_model.maker_bps = info.maker_bps
            if info.taker_bps is not None:
                self._fee_model.taker_bps = info.taker_bps

        return info.to_fee_overrides()

    def update_rates(
        self,
        maker_bps: Optional[float] = None,
        taker_bps: Optional[float] = None,
        vip_tier: Optional[int] = None,
    ) -> None:
        """
        Update fee rates.

        Args:
            maker_bps: New maker fee rate
            taker_bps: New taker fee rate
            vip_tier: New VIP tier
        """
        model = self._get_fee_model()

        if maker_bps is not None:
            model.maker_bps = float(maker_bps)
        if taker_bps is not None:
            model.taker_bps = float(taker_bps)
        if vip_tier is not None:
            model.vip_tier = int(vip_tier)
