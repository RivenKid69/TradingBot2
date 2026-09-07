# -*- coding: utf-8 -*-
"""
execution_providers_futures.py
L2 Execution Providers for Futures Markets (Phase 4A).

Extends crypto parametric TCA with futures-specific factors:
1. Funding rate impact on spread (crypto perpetuals)
2. Open interest impact on liquidity
3. Liquidation cascade simulation
4. Mark vs last price execution

Supports:
- Crypto Perpetual (Binance USDT-M)
- Crypto Quarterly (Binance delivery)

References:
- Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
- Cont et al. (2014): "The Price Impact of Order Book Events"
- Binance USDT-M Futures Documentation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, Dict, Any, Protocol
import logging
import math

from execution_providers import (
    SlippageProvider,
    FeeProvider,
    FillProvider,
    CryptoParametricSlippageProvider,
    CryptoParametricConfig,
    OHLCVFillProvider,
    MarketState,
    Order,
    BarData,
    Fill,
    VolatilityRegime,
)


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class FuturesSlippageConfig(CryptoParametricConfig):
    """
    Configuration for FuturesSlippageProvider.

    Extends CryptoParametricConfig with futures-specific parameters.

    Attributes:
        funding_impact_sensitivity: Multiplier for funding rate stress
            - Default: 5.0 (0.1% funding = +5bps slippage)
            - High sensitivity for crowded positions

        oi_concentration_threshold: Open interest concentration warning level
            - Default: 0.3 (30% of total OI triggers warning)
            - Used to detect market manipulation risk

        liquidation_cascade_sensitivity: Liquidation volume impact multiplier
            - Default: 5.0 (1% liquidations = +5bps slippage)
            - Models cascading liquidations effect

        liquidation_cascade_threshold: Minimum liquidation ratio to trigger cascade
            - Default: 0.01 (1% of ADV)
            - Below this, liquidations are noise

        open_interest_liquidity_factor: OI impact on liquidity
            - Default: 0.1 (10% OI increase = +1bps slippage)
            - Higher OI = more crowded = less liquid

        use_mark_price_execution: Execute at mark price instead of last
            - Default: True for futures (prevents manipulation)
            - Mark price = TWAP of index + funding basis
    """

    # Futures-specific parameters
    funding_impact_sensitivity: float = 5.0
    oi_concentration_threshold: float = 0.3
    liquidation_cascade_sensitivity: float = 5.0
    liquidation_cascade_threshold: float = 0.01
    liquidation_cascade_max_factor: float = 3.0  # Cap at 200% increase
    open_interest_liquidity_factor: float = 0.1
    open_interest_max_penalty: float = 2.0  # Cap at 100% increase
    use_mark_price_execution: bool = True


# ═══════════════════════════════════════════════════════════════════════════
# SLIPPAGE PROVIDER
# ═══════════════════════════════════════════════════════════════════════════


class FuturesSlippageProvider(CryptoParametricSlippageProvider):
    """
    L2+ Futures-specific slippage model.

    Extends CryptoParametricSlippageProvider with:
    - Funding rate stress (high funding = directional pressure)
    - Open interest concentration (crowded positions)
    - Liquidation cascade risk (forced selling)

    Total Slippage Formula:
        slippage = base_slippage
            × funding_stress        # Crowded position penalty
            × liquidation_cascade   # Forced liquidation impact
            × oi_liquidity_penalty  # Concentration risk

    Example:
        >>> config = FuturesSlippageConfig(funding_impact_sensitivity=5.0)
        >>> provider = FuturesSlippageProvider(config=config)
        >>>
        >>> # High positive funding + BUY = extra cost (crowded long)
        >>> slippage = provider.compute_slippage_bps(
        ...     order=Order("BTCUSDT", "BUY", Decimal("0.1"), "MARKET"),
        ...     market=MarketState(0, Decimal("50000"), Decimal("50001"), adv=1e9),
        ...     participation_ratio=0.001,
        ...     funding_rate=0.001,  # 0.1% = very high
        ...     open_interest=50_000_000,
        ...     recent_liquidations=1_000_000,
        ... )
        >>> # slippage > base_slippage due to funding stress + liquidations

    References:
        - Binance Futures Funding Rate Mechanism
        - Zhao et al. (2020): "Liquidation Cascade Effects in Crypto Markets"
    """

    def __init__(
        self,
        config: Optional[FuturesSlippageConfig] = None,
        **kwargs,
    ):
        """
        Initialize futures slippage provider.

        Args:
            config: Futures-specific configuration
            **kwargs: Forwarded to parent CryptoParametricSlippageProvider
        """
        self.futures_config = config or FuturesSlippageConfig()
        super().__init__(config=self.futures_config, **kwargs)

    def compute_slippage_bps(
        self,
        order: Order,
        market: MarketState,
        participation_ratio: float,
        funding_rate: Optional[float] = None,
        open_interest: Optional[float] = None,
        recent_liquidations: Optional[float] = None,
        **kwargs,
    ) -> float:
        """
        Compute futures slippage with additional factors.

        Args:
            order: Order to execute
            market: Current market state
            participation_ratio: Order size / ADV
            funding_rate: Current funding rate (e.g., 0.0001 = 0.01%)
            open_interest: Total open interest in notional
            recent_liquidations: Liquidation volume in recent period (e.g., 1h)
            **kwargs: Additional parameters for parent class

        Returns:
            Expected slippage in basis points (always non-negative)

        Notes:
            - Funding rate in same direction as order increases slippage
            - High open interest concentration adds liquidity penalty
            - Recent liquidations trigger cascade factor
        """
        # =====================================================================
        # 1. Base slippage from crypto model
        # =====================================================================
        base_bps = super().compute_slippage_bps(
            order=order,
            market=market,
            participation_ratio=participation_ratio,
            funding_rate=funding_rate,  # Already used by parent
            **kwargs,
        )

        # =====================================================================
        # 2. Funding stress factor
        # =====================================================================
        # High positive funding + buying = extra cost (crowded long)
        # High negative funding + selling = extra cost (crowded short)
        funding_stress = 1.0
        if funding_rate is not None:
            is_same_direction = (funding_rate > 0 and str(order.side).upper() == "BUY") or (
                funding_rate < 0 and str(order.side).upper() == "SELL"
            )
            if is_same_direction:
                # FIX (2025-12-02): Use multiplicative ratio, not additive bps
                # E.g., 0.0001 (0.01%) × 5.0 = 0.0005 = 0.05% increase
                # Original had × 10000 which gave 51x for 0.1% funding (WRONG!)
                funding_stress = (
                    1.0 + abs(funding_rate) * self.futures_config.funding_impact_sensitivity
                )

        # =====================================================================
        # 3. Liquidation cascade factor
        # =====================================================================
        cascade_factor = 1.0
        if recent_liquidations is not None and market.adv is not None and market.adv > 0:
            liquidation_ratio = recent_liquidations / market.adv

            # Only trigger if above threshold
            if liquidation_ratio > self.futures_config.liquidation_cascade_threshold:
                # Scale by sensitivity: 1% liquidations × 5.0 = +5% slippage
                # FIX (2025-12-02): Cap at max_factor to prevent extreme scenarios
                cascade_factor = min(
                    self.futures_config.liquidation_cascade_max_factor,
                    1.0 + liquidation_ratio * self.futures_config.liquidation_cascade_sensitivity,
                )

        # =====================================================================
        # 4. Open interest liquidity penalty
        # =====================================================================
        # High OI relative to ADV means crowded position
        oi_penalty = 1.0
        if open_interest is not None and market.adv is not None and market.adv > 0:
            oi_to_adv = open_interest / market.adv

            # Penalty grows with OI concentration
            # E.g., OI = 3× ADV → penalty = 1.0 + 3.0 × 0.1 = 1.3 (+30%)
            # FIX (2025-12-02): Cap at max_penalty to prevent extreme scenarios
            if oi_to_adv > 1.0:
                oi_penalty = min(
                    self.futures_config.open_interest_max_penalty,
                    1.0 + (oi_to_adv - 1.0) * self.futures_config.open_interest_liquidity_factor,
                )

        # =====================================================================
        # 5. Combine all factors
        # =====================================================================
        total_slippage = base_bps * funding_stress * cascade_factor * oi_penalty

        # Apply bounds from config
        total_slippage = max(self.futures_config.min_slippage_bps, total_slippage)
        total_slippage = min(self.futures_config.max_slippage_bps, total_slippage)

        return float(total_slippage)

    def estimate_liquidation_risk(
        self,
        order: Order,
        market: MarketState,
        position_size: float,
        entry_price: float,
        leverage: float,
        maintenance_margin_rate: float = 0.004,  # 0.4% for BTC perpetual
    ) -> Dict[str, Any]:
        """
        Estimate liquidation risk for a position.

        Args:
            order: Proposed order
            market: Current market state
            position_size: Current position size (negative for short)
            entry_price: Position entry price
            leverage: Position leverage (e.g., 10.0 for 10x)
            maintenance_margin_rate: Maintenance margin rate (default 0.4%)

        Returns:
            Dict with liquidation analysis:
                - liquidation_price: Price at which position is liquidated
                - distance_to_liquidation_bps: Distance in basis points
                - is_high_risk: True if <500bps from liquidation

        Formula (long position):
            liquidation_price = entry_price × (1 - 1/leverage + MM%)

        Formula (short position):
            liquidation_price = entry_price × (1 + 1/leverage - MM%)
        """
        is_long = position_size > 0

        # Calculate liquidation price
        if is_long:
            liquidation_price = entry_price * (1 - 1 / leverage + maintenance_margin_rate)
        else:
            liquidation_price = entry_price * (1 + 1 / leverage - maintenance_margin_rate)

        # Distance to liquidation
        current_price = float(market.get_mid_price())
        if is_long:
            distance_bps = ((current_price - liquidation_price) / current_price) * 10000
        else:
            distance_bps = ((liquidation_price - current_price) / current_price) * 10000

        # Risk assessment
        is_high_risk = distance_bps < 500  # Within 5% of liquidation

        return {
            "liquidation_price": liquidation_price,
            "distance_to_liquidation_bps": distance_bps,
            "is_high_risk": is_high_risk,
            "current_price": current_price,
            "leverage": leverage,
        }


# ═══════════════════════════════════════════════════════════════════════════
# FEE PROVIDER
# ═══════════════════════════════════════════════════════════════════════════


class FuturesFeeProvider:
    """
    Futures fee provider.

    Includes:
    - Maker/taker fees (same structure as spot)
    - Funding payments (tracked separately, not part of execution fee)
    - Liquidation fees (0.5% on Binance)

    Fee Structure (Binance USDT-M):
        - Maker: 0.02% (2 bps)
        - Taker: 0.04% (4 bps)
        - Liquidation: 0.50% (50 bps)

    Example:
        >>> provider = FuturesFeeProvider()
        >>>
        >>> # Normal trade
        >>> fee = provider.compute_fee(fill, is_liquidation=False)
        >>> # fee = notional × (2bps for maker or 4bps for taker)
        >>>
        >>> # Liquidation
        >>> liq_fee = provider.compute_fee(fill, is_liquidation=True)
        >>> # liq_fee = notional × 50bps

    Notes:
        - Funding payments are handled separately (not execution fees)
        - Liquidation fees go to insurance fund
        - VIP tiers can reduce maker/taker fees (not modeled here)
    """

    def __init__(
        self,
        maker_bps: float = 2.0,
        taker_bps: float = 4.0,
        liquidation_fee_bps: float = 50.0,
    ):
        """
        Initialize futures fee provider.

        Args:
            maker_bps: Maker fee in basis points (default: 2bps = 0.02%)
            taker_bps: Taker fee in basis points (default: 4bps = 0.04%)
            liquidation_fee_bps: Liquidation fee in basis points (default: 50bps = 0.5%)
        """
        self._maker_bps = maker_bps
        self._taker_bps = taker_bps
        self._liquidation_fee_bps = liquidation_fee_bps

    def compute_fee(
        self,
        fill: Fill,
        is_liquidation: bool = False,
    ) -> Decimal:
        """
        Compute fee for fill.

        Args:
            fill: Executed fill
            is_liquidation: True if this is a liquidation order

        Returns:
            Fee amount in quote currency (USDT)
        """
        if is_liquidation:
            # Liquidation fee: 0.5% of notional
            return fill.notional * Decimal(str(self._liquidation_fee_bps)) / Decimal("10000")

        # Regular trade: maker or taker
        # Check liquidity field (string "maker" or "taker") or metadata for is_maker
        is_maker = False
        if hasattr(fill, "liquidity") and fill.liquidity == "maker":
            is_maker = True
        elif hasattr(fill, "metadata") and fill.metadata:
            is_maker = fill.metadata.get("is_maker", False)
        elif hasattr(fill, "is_maker"):
            is_maker = fill.is_maker

        bps = self._maker_bps if is_maker else self._taker_bps

        # Get notional, with fallback to price * qty if notional is None
        notional = fill.notional
        if notional is None and hasattr(fill, "price") and hasattr(fill, "qty"):
            notional = fill.price * fill.qty
        if notional is None:
            notional = 0

        return float(notional) * float(bps) / 10000.0

    def compute_funding_payment(
        self,
        position_size: Decimal,
        mark_price: Decimal,
        funding_rate: Decimal,
    ) -> Decimal:
        """
        Compute funding payment for position.

        Formula:
            funding_payment = position_notional × funding_rate

        - Positive funding: Longs pay shorts
        - Negative funding: Shorts pay longs

        Args:
            position_size: Position size (positive for long, negative for short)
            mark_price: Current mark price
            funding_rate: Current funding rate (e.g., 0.0001 = 0.01%)

        Returns:
            Funding payment (positive = receive, negative = pay)

        Example:
            >>> # Long 1 BTC at $50,000, funding = +0.01%
            >>> payment = provider.compute_funding_payment(
            ...     Decimal("1.0"),
            ...     Decimal("50000"),
            ...     Decimal("0.0001"),
            ... )
            >>> # payment = -$5 (long pays)
            >>>
            >>> # Short 1 BTC at $50,000, funding = +0.01%
            >>> payment = provider.compute_funding_payment(
            ...     Decimal("-1.0"),
            ...     Decimal("50000"),
            ...     Decimal("0.0001"),
            ... )
            >>> # payment = +$5 (short receives)
        """
        position_notional = abs(position_size) * mark_price
        payment = position_notional * funding_rate

        # Long position pays when funding is positive
        if position_size > 0:
            return -payment
        else:
            return payment


# ═══════════════════════════════════════════════════════════════════════════
# L2 EXECUTION PROVIDER
# ═══════════════════════════════════════════════════════════════════════════


class FuturesL2ExecutionProvider:
    """
    L2 execution provider for futures.

    Combines:
    - FuturesSlippageProvider (with funding/OI/liquidation factors)
    - FuturesFeeProvider (with liquidation fees)
    - OHLCVFillProvider (can use mark price bars)

    Features:
    - Mark price execution (prevents manipulation)
    - Funding rate impact on slippage
    - Liquidation cascade detection
    - Open interest liquidity adjustment

    Example:
        >>> provider = FuturesL2ExecutionProvider(use_mark_price=True)
        >>>
        >>> fill = provider.execute(
        ...     order=Order("BTCUSDT", "BUY", Decimal("0.1"), "MARKET"),
        ...     market=MarketState(...),
        ...     bar=BarData(...),
        ...     funding_rate=0.0001,
        ...     mark_bar=BarData(...),  # Mark price bar
        ... )
        >>>
        >>> # fill.price uses mark_bar (if provided)
        >>> # fill.slippage_bps includes funding stress
        >>> # fill.fee includes maker/taker fee
    """

    def __init__(
        self,
        use_mark_price: bool = True,
        slippage_config: Optional[FuturesSlippageConfig] = None,
        **kwargs,
    ):
        """
        Initialize futures L2 execution provider.

        Args:
            use_mark_price: Use mark price for execution (default: True)
            slippage_config: Slippage configuration
            **kwargs: Additional parameters for components
        """
        self._use_mark_price = use_mark_price
        self._slippage = FuturesSlippageProvider(config=slippage_config, **kwargs)
        self._fees = FuturesFeeProvider(**kwargs)
        self._fill = OHLCVFillProvider(**kwargs)

    def execute(
        self,
        order: Order,
        market: MarketState,
        bar: BarData,
        funding_rate: Optional[float] = None,
        mark_bar: Optional[BarData] = None,
        open_interest: Optional[float] = None,
        recent_liquidations: Optional[float] = None,
    ) -> Fill:
        """
        Execute order with futures mechanics.

        Args:
            order: Order to execute
            market: Current market state
            bar: Last price OHLCV bar
            funding_rate: Current funding rate (optional)
            mark_bar: Mark price OHLCV bar (optional, used if use_mark_price=True)
            open_interest: Total open interest (optional)
            recent_liquidations: Recent liquidation volume (optional)

        Returns:
            Fill with futures-specific adjustments

        Notes:
            - If mark_bar provided and use_mark_price=True: execution uses mark price
            - Slippage includes funding stress + liquidation cascade + OI penalty
            - Fees use maker/taker structure (same as spot)
        """
        # Use mark price bar if available and configured
        exec_bar = mark_bar if (self._use_mark_price and mark_bar) else bar

        # Compute fill using selected bar
        fill = self._fill.try_fill(order, market, exec_bar)

        if fill is None or not getattr(fill, "is_filled", False):
            return fill

        # Compute participation ratio
        if market.adv and market.adv > 0:
            participation_ratio = float(fill.notional) / market.adv
        else:
            participation_ratio = 0.0

        # Apply futures-specific slippage
        slippage_bps = self._slippage.compute_slippage_bps(
            order=order,
            market=market,
            participation_ratio=participation_ratio,
            funding_rate=funding_rate,
            open_interest=open_interest,
            recent_liquidations=recent_liquidations,
        )

        # Adjust fill price with slippage
        # (assuming Fill has a method to adjust price)
        # For now, we'll create a new fill with adjusted values
        slippage_factor = 1.0 + (slippage_bps / 10000.0)
        if str(order.side).upper() == "BUY":
            adjusted_price = fill.price * Decimal(str(slippage_factor))
        else:
            adjusted_price = fill.price / Decimal(str(slippage_factor))

        # Recompute notional with adjusted price
        adjusted_notional = fill.qty * adjusted_price

        # Compute fees
        fee = self._fees.compute_fee(fill, is_liquidation=False)

        # Return adjusted fill
        # Note: This assumes Fill class supports these attributes
        # In real implementation, use Fill's methods or create new Fill
        return Fill(
            symbol=fill.symbol,
            side=fill.side,
            qty=fill.qty,
            price=adjusted_price,
            notional=adjusted_notional,
            is_filled=True,
            is_maker=fill.is_maker,
            fee=fee,
            slippage_bps=slippage_bps,
        )


# ═══════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def create_futures_slippage_provider(
    config: Optional[FuturesSlippageConfig] = None,
) -> FuturesSlippageProvider:
    """
    Factory function to create futures slippage provider.

    Args:
        config: Optional configuration

    Returns:
        FuturesSlippageProvider instance
    """
    return FuturesSlippageProvider(config=config)


def create_futures_execution_provider(
    use_mark_price: bool = True,
    slippage_config: Optional[FuturesSlippageConfig] = None,
) -> FuturesL2ExecutionProvider:
    """
    Factory function to create L2 futures execution provider.

    Args:
        use_mark_price: Use mark price for execution
        slippage_config: Slippage configuration

    Returns:
        FuturesL2ExecutionProvider instance
    """
    return FuturesL2ExecutionProvider(
        use_mark_price=use_mark_price,
        slippage_config=slippage_config,
    )
