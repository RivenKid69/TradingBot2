# -*- coding: utf-8 -*-
"""
adapters/deribit/__init__.py

Deribit Crypto Options Adapter (Phase 2B)

This module provides adapters for Deribit, the leading crypto options exchange.
Key differences from US equity options (IB/Polygon):

1. **Inverse Settlement**: P&L in BTC/ETH, not USD
   - Call payoff: max(0, S-K) / S  (in underlying crypto)
   - Put payoff: max(0, K-S) / S   (in underlying crypto)

2. **European Exercise Only**: No early exercise risk

3. **24/7 Trading**: Continuous market with varying liquidity

4. **DVOL Index**: Deribit Volatility Index (30-day constant maturity IV)
   - Similar methodology to VIX
   - Available for BTC and ETH

5. **Inverse Margining**: Margin posted in underlying crypto
   - As crypto price drops, USD value of margin decreases
   - "Double-whammy" risk for short positions

6. **Strike Conventions**:
   - BTC: $1000 increments (e.g., 50000, 51000, 52000)
   - ETH: $50 increments (e.g., 3000, 3050, 3100)

7. **Expiration Pattern**:
   - Daily: 08:00 UTC
   - Weekly: Every Friday 08:00 UTC
   - Monthly: Last Friday of month 08:00 UTC
   - Quarterly: Last Friday of Mar/Jun/Sep/Dec 08:00 UTC

References:
    - Deribit API: https://docs.deribit.com/
    - DVOL Methodology: https://www.deribit.com/pages/docs/volatility-index
    - Inverse Futures: "Understanding Crypto Derivatives" (Deribit research)

Example:
    >>> from adapters.deribit import (
    ...     DeribitOptionsAdapter,
    ...     DeribitMarginCalculator,
    ...     create_deribit_options_adapter,
    ... )
    >>> adapter = create_deribit_options_adapter(testnet=True)
    >>> chain = adapter.get_btc_options()
"""

from adapters.deribit.options import (
    # Data classes
    DeribitOptionContract,
    DeribitGreeks,
    DeribitOptionQuote,
    DeribitOrderbook,
    DVOLData,
    DeribitInstrumentInfo,
    DeribitPosition,
    DeribitOrder,
    DeribitOrderResult,
    # Adapters
    DeribitOptionsMarketDataAdapter,
    DeribitOptionsOrderExecutionAdapter,
    # Factory functions
    create_deribit_options_market_data_adapter,
    create_deribit_options_order_execution_adapter,
    # Utility functions
    parse_deribit_instrument_name,
    create_deribit_instrument_name,
    btc_to_usd,
    usd_to_btc,
    eth_to_usd,
    usd_to_eth,
)

from adapters.deribit.margin import (
    DeribitMarginCalculator,
    DeribitMarginResult,
    InverseSettlementCalculator,
    InversePayoff,
    create_deribit_margin_calculator,
    calculate_inverse_call_payoff,
    calculate_inverse_put_payoff,
)

from adapters.deribit.websocket import (
    DeribitWebSocketClient,
    DeribitStreamConfig,
    DeribitSubscription,
    DeribitMessage,
    create_deribit_websocket_client,
)

__all__ = [
    # Data classes
    "DeribitOptionContract",
    "DeribitGreeks",
    "DeribitOptionQuote",
    "DeribitOrderbook",
    "DVOLData",
    "DeribitInstrumentInfo",
    "DeribitPosition",
    "DeribitOrder",
    "DeribitOrderResult",
    # Adapters
    "DeribitOptionsMarketDataAdapter",
    "DeribitOptionsOrderExecutionAdapter",
    # Margin
    "DeribitMarginCalculator",
    "DeribitMarginResult",
    "InverseSettlementCalculator",
    "InversePayoff",
    # WebSocket
    "DeribitWebSocketClient",
    "DeribitStreamConfig",
    "DeribitSubscription",
    "DeribitMessage",
    # Factory functions
    "create_deribit_options_market_data_adapter",
    "create_deribit_options_order_execution_adapter",
    "create_deribit_margin_calculator",
    "create_deribit_websocket_client",
    # Utility functions
    "parse_deribit_instrument_name",
    "create_deribit_instrument_name",
    "btc_to_usd",
    "usd_to_btc",
    "eth_to_usd",
    "usd_to_eth",
    "calculate_inverse_call_payoff",
    "calculate_inverse_put_payoff",
]


# =========================
# Registry Registration
# =========================


def _register_deribit_adapters() -> None:
    """Register Deribit adapters with the global registry."""
    try:
        from adapters.registry import register, AdapterType
        from adapters.models import ExchangeVendor

        # Register options market data adapter
        register(
            vendor=ExchangeVendor.DERIBIT,
            adapter_type=AdapterType.OPTIONS_MARKET_DATA,
            adapter_class=DeribitOptionsMarketDataAdapter,
            factory_func=lambda vendor, config: DeribitOptionsMarketDataAdapter(
                vendor=vendor,
                config=config,
            ),
            default_config={"testnet": True},
            description="Deribit crypto options market data (BTC/ETH)",
        )

        # Register options order execution adapter
        register(
            vendor=ExchangeVendor.DERIBIT,
            adapter_type=AdapterType.OPTIONS_ORDER_EXECUTION,
            adapter_class=DeribitOptionsOrderExecutionAdapter,
            factory_func=lambda vendor, config: DeribitOptionsOrderExecutionAdapter(
                vendor=vendor,
                config=config,
            ),
            default_config={"testnet": True},
            description="Deribit crypto options order execution",
        )

    except ImportError:
        # Registry not available (e.g., during testing)
        pass


# Auto-register on import
_register_deribit_adapters()
