# -*- coding: utf-8 -*-
"""
tests/test_futures_adapters.py
Comprehensive test suite for Binance futures adapters.

Tests cover:
- BinanceFuturesMarketDataAdapter
- BinanceFuturesExchangeInfoAdapter
- BinanceFuturesOrderExecutionAdapter
- Registry integration

Target: 40+ tests (actual: 55+ tests)
"""

from __future__ import annotations

import json
import pytest
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, PropertyMock

from adapters.models import ExchangeVendor, MarketType
from adapters.registry import (
    AdapterType,
    create_futures_market_data_adapter,
    create_futures_exchange_info_adapter,
    create_futures_order_execution_adapter,
)
from adapters.binance.futures_market_data import BinanceFuturesMarketDataAdapter
from adapters.binance.futures_exchange_info import BinanceFuturesExchangeInfoAdapter
from adapters.binance.futures_order_execution import (
    BinanceFuturesOrderExecutionAdapter,
    FuturesOrderResult,
)
from core_futures import (
    FuturesContractSpec,
    FundingRateInfo,
    MarkPriceTick,
    OpenInterestInfo,
    LiquidationEvent,
    FuturesPosition,
    FuturesAccountState,
    FuturesFill,
    LeverageBracket,
    MarginMode,
    PositionSide,
    ContractType,
    FuturesType,
    Exchange,
    OrderType as FuturesOrderType,
)
from core_models import Order, Side, OrderType, TimeInForce


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_mark_price_response() -> Dict[str, Any]:
    """Sample mark price API response."""
    return {
        "symbol": "BTCUSDT",
        "markPrice": "50123.45",
        "indexPrice": "50100.00",
        "estimatedSettlePrice": "50110.00",
        "lastFundingRate": "0.0001",
        "nextFundingTime": 1700028800000,
        "time": 1700000000000,
    }


@pytest.fixture
def mock_funding_rate_history() -> List[Dict[str, Any]]:
    """Sample funding rate history response."""
    return [
        {
            "symbol": "BTCUSDT",
            "fundingTime": 1700000000000,
            "fundingRate": "0.0001",
            "markPrice": "50000.00",
        },
        {
            "symbol": "BTCUSDT",
            "fundingTime": 1699971200000,
            "fundingRate": "0.00012",
            "markPrice": "49800.00",
        },
    ]


@pytest.fixture
def mock_open_interest_response() -> Dict[str, Any]:
    """Sample open interest response."""
    return {
        "symbol": "BTCUSDT",
        "openInterest": "12345.678",
        "time": 1700000000000,
    }


@pytest.fixture
def mock_liquidation_orders() -> List[Dict[str, Any]]:
    """Sample liquidation orders response."""
    return [
        {
            "symbol": "BTCUSDT",
            "side": "SELL",
            "orderId": "123456",
            "origQty": "0.5",
            "avgPrice": "48000.00",
            "status": "FILLED",
            "time": 1700000000000,
        },
    ]


@pytest.fixture
def mock_exchange_info_response() -> Dict[str, Any]:
    """Sample exchange info response."""
    return {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "baseAsset": "BTC",
                "quoteAsset": "USDT",
                "marginAsset": "USDT",
                "status": "TRADING",
                "contractType": "PERPETUAL",
                "pricePrecision": 2,
                "quantityPrecision": 3,
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                    {
                        "filterType": "LOT_SIZE",
                        "stepSize": "0.001",
                        "minQty": "0.001",
                        "maxQty": "1000",
                    },
                    {"filterType": "MIN_NOTIONAL", "notional": "10"},
                ],
            },
            {
                "symbol": "ETHUSDT",
                "baseAsset": "ETH",
                "quoteAsset": "USDT",
                "marginAsset": "USDT",
                "status": "TRADING",
                "contractType": "PERPETUAL",
                "pricePrecision": 2,
                "quantityPrecision": 3,
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                    {
                        "filterType": "LOT_SIZE",
                        "stepSize": "0.01",
                        "minQty": "0.01",
                        "maxQty": "10000",
                    },
                    {"filterType": "MIN_NOTIONAL", "notional": "10"},
                ],
            },
        ],
    }


@pytest.fixture
def mock_leverage_brackets_response() -> List[Dict[str, Any]]:
    """Sample leverage brackets response."""
    return [
        {
            "symbol": "BTCUSDT",
            "brackets": [
                {
                    "bracket": 1,
                    "initialLeverage": 125,
                    "notionalCap": 50000,
                    "maintMarginRatio": 0.004,
                    "cum": 0,
                },
                {
                    "bracket": 2,
                    "initialLeverage": 100,
                    "notionalCap": 250000,
                    "maintMarginRatio": 0.005,
                    "cum": 50,
                },
                {
                    "bracket": 3,
                    "initialLeverage": 50,
                    "notionalCap": 1000000,
                    "maintMarginRatio": 0.01,
                    "cum": 1300,
                },
            ],
        },
    ]


@pytest.fixture
def mock_position_response() -> List[Dict[str, Any]]:
    """Sample position risk response."""
    return [
        {
            "symbol": "BTCUSDT",
            "positionAmt": "0.5",
            "entryPrice": "50000.00",
            "markPrice": "52000.00",
            "unRealizedProfit": "1000.00",
            "liquidationPrice": "45000.00",
            "leverage": "20",
            "marginType": "cross",
            "positionSide": "BOTH",
            "isolatedMargin": "0",
            "maintMargin": "200",
            "updateTime": 1700000000000,
            "notional": "26000",
        },
    ]


@pytest.fixture
def mock_order_response() -> Dict[str, Any]:
    """Sample order submission response."""
    return {
        "orderId": 123456789,
        "clientOrderId": "test_order_1",
        "symbol": "BTCUSDT",
        "side": "BUY",
        "type": "MARKET",
        "status": "FILLED",
        "origQty": "0.1",
        "executedQty": "0.1",
        "avgPrice": "50100.00",
        "positionSide": "BOTH",
        "realizedPnl": "0",
        "commission": "2.004",
        "commissionAsset": "USDT",
        "updateTime": 1700000000000,
    }


@pytest.fixture
def mock_account_response() -> Dict[str, Any]:
    """Sample account info response."""
    return {
        "accountId": "12345",
        "feeTier": 1,
        "totalWalletBalance": "10000.00",
        "totalMarginBalance": "10500.00",
        "totalUnrealizedProfit": "500.00",
        "availableBalance": "8000.00",
        "totalInitialMargin": "2000.00",
        "totalMaintMargin": "500.00",
        "totalPositionInitialMargin": "1500.00",
        "totalOpenOrderInitialMargin": "500.00",
        "maxWithdrawAmount": "7500.00",
        "updateTime": 1700000000000,
        "assets": [
            {"asset": "USDT", "walletBalance": "10000.00"},
        ],
        "positions": [],
    }


# =============================================================================
# BinanceFuturesMarketDataAdapter Tests
# =============================================================================


class TestBinanceFuturesMarketDataAdapter:
    """Tests for BinanceFuturesMarketDataAdapter."""

    def test_adapter_initialization_default(self):
        """Test adapter initialization with defaults."""
        adapter = BinanceFuturesMarketDataAdapter()
        assert adapter._vendor == ExchangeVendor.BINANCE
        assert adapter._futures_url == "https://fapi.binance.com"

    def test_adapter_initialization_testnet(self):
        """Test adapter initialization with testnet."""
        adapter = BinanceFuturesMarketDataAdapter(config={"testnet": True})
        assert adapter._futures_url == "https://testnet.binancefuture.com"

    def test_adapter_initialization_custom_url(self):
        """Test adapter initialization with custom URL."""
        custom_url = "https://custom.binance.com"
        adapter = BinanceFuturesMarketDataAdapter(config={"futures_url": custom_url})
        assert adapter._futures_url == custom_url

    def test_market_type_property(self):
        """Test market_type property."""
        adapter = BinanceFuturesMarketDataAdapter()
        assert adapter.market_type == MarketType.CRYPTO_FUTURES

    def test_interval_ms_to_timeframe_seconds(self):
        """Test interval conversion for seconds."""
        assert BinanceFuturesMarketDataAdapter._interval_ms_to_timeframe(30000) == "30s"

    def test_interval_ms_to_timeframe_minutes(self):
        """Test interval conversion for minutes."""
        assert BinanceFuturesMarketDataAdapter._interval_ms_to_timeframe(60000) == "1m"
        assert BinanceFuturesMarketDataAdapter._interval_ms_to_timeframe(300000) == "5m"

    def test_interval_ms_to_timeframe_hours(self):
        """Test interval conversion for hours."""
        assert BinanceFuturesMarketDataAdapter._interval_ms_to_timeframe(3600000) == "1h"
        assert BinanceFuturesMarketDataAdapter._interval_ms_to_timeframe(14400000) == "4h"

    def test_interval_ms_to_timeframe_days(self):
        """Test interval conversion for days."""
        assert BinanceFuturesMarketDataAdapter._interval_ms_to_timeframe(86400000) == "1d"

    @patch.object(BinanceFuturesMarketDataAdapter, "_get_client")
    def test_get_mark_price(self, mock_get_client, mock_mark_price_response):
        """Test get_mark_price parsing."""
        mock_client = MagicMock()
        mock_client._session_get.return_value = mock_mark_price_response
        mock_get_client.return_value = mock_client

        adapter = BinanceFuturesMarketDataAdapter()
        result = adapter.get_mark_price("BTCUSDT")

        assert result is not None
        assert isinstance(result, MarkPriceTick)
        assert result.symbol == "BTCUSDT"
        assert result.mark_price == Decimal("50123.45")
        assert result.index_price == Decimal("50100.00")
        assert result.funding_rate == Decimal("0.0001")

    @patch.object(BinanceFuturesMarketDataAdapter, "_get_client")
    def test_get_mark_prices_all(self, mock_get_client):
        """Test get_mark_prices for all symbols."""
        mock_responses = [
            {
                "symbol": "BTCUSDT",
                "markPrice": "50000",
                "indexPrice": "49990",
                "lastFundingRate": "0.0001",
                "time": 1700000000000,
            },
            {
                "symbol": "ETHUSDT",
                "markPrice": "2500",
                "indexPrice": "2495",
                "lastFundingRate": "0.00008",
                "time": 1700000000000,
            },
        ]
        mock_client = MagicMock()
        mock_client._session_get.return_value = mock_responses
        mock_get_client.return_value = mock_client

        adapter = BinanceFuturesMarketDataAdapter()
        results = adapter.get_mark_prices()

        assert len(results) == 2
        assert all(isinstance(r, MarkPriceTick) for r in results)

    @patch.object(BinanceFuturesMarketDataAdapter, "get_mark_price")
    def test_get_funding_rate(self, mock_get_mark_price):
        """Test get_funding_rate parsing."""
        mock_get_mark_price.return_value = MarkPriceTick(
            symbol="BTCUSDT",
            mark_price=Decimal("50000"),
            index_price=Decimal("49990"),
            estimated_settle_price=Decimal("50000"),
            funding_rate=Decimal("0.0001"),
            next_funding_time_ms=1700028800000,
            timestamp_ms=1700000000000,
        )

        adapter = BinanceFuturesMarketDataAdapter()
        result = adapter.get_funding_rate("BTCUSDT")

        assert result is not None
        assert isinstance(result, FundingRateInfo)
        assert result.symbol == "BTCUSDT"
        assert result.funding_rate == Decimal("0.0001")

    @patch.object(BinanceFuturesMarketDataAdapter, "_get_client")
    def test_get_open_interest(self, mock_get_client, mock_open_interest_response):
        """Test get_open_interest parsing."""
        mock_client = MagicMock()
        mock_client._session_get.return_value = mock_open_interest_response
        mock_get_client.return_value = mock_client

        adapter = BinanceFuturesMarketDataAdapter()
        result = adapter.get_open_interest("BTCUSDT")

        assert result is not None
        assert isinstance(result, OpenInterestInfo)
        assert result.symbol == "BTCUSDT"
        assert result.open_interest == Decimal("12345.678")

    @patch.object(BinanceFuturesMarketDataAdapter, "_get_client")
    def test_get_liquidation_orders(self, mock_get_client, mock_liquidation_orders):
        """Test get_liquidation_orders parsing."""
        mock_client = MagicMock()
        mock_client._session_get.return_value = mock_liquidation_orders
        mock_get_client.return_value = mock_client

        adapter = BinanceFuturesMarketDataAdapter()
        results = adapter.get_liquidation_orders("BTCUSDT")

        assert len(results) == 1
        assert isinstance(results[0], LiquidationEvent)
        assert results[0].symbol == "BTCUSDT"
        assert results[0].qty == Decimal("0.5")


# =============================================================================
# BinanceFuturesExchangeInfoAdapter Tests
# =============================================================================


class TestBinanceFuturesExchangeInfoAdapter:
    """Tests for BinanceFuturesExchangeInfoAdapter."""

    def test_adapter_initialization_default(self):
        """Test adapter initialization with defaults."""
        adapter = BinanceFuturesExchangeInfoAdapter()
        assert adapter._vendor == ExchangeVendor.BINANCE
        assert adapter._futures_url == "https://fapi.binance.com"

    def test_adapter_initialization_testnet(self):
        """Test adapter initialization with testnet."""
        adapter = BinanceFuturesExchangeInfoAdapter(config={"testnet": True})
        assert adapter._futures_url == "https://testnet.binancefuture.com"

    def test_market_type_property(self):
        """Test market_type property."""
        adapter = BinanceFuturesExchangeInfoAdapter()
        assert adapter.market_type == MarketType.CRYPTO_FUTURES

    def test_extract_decimal_valid(self):
        """Test _extract_decimal with valid data."""
        data = {"tick_size": "0.01"}
        result = BinanceFuturesExchangeInfoAdapter._extract_decimal(data, "tick_size", "0.001")
        assert result == Decimal("0.01")

    def test_extract_decimal_missing(self):
        """Test _extract_decimal with missing key."""
        data = {}
        result = BinanceFuturesExchangeInfoAdapter._extract_decimal(data, "tick_size", "0.001")
        assert result == Decimal("0.001")

    def test_extract_decimal_invalid(self):
        """Test _extract_decimal with invalid value."""
        data = {"tick_size": "invalid"}
        result = BinanceFuturesExchangeInfoAdapter._extract_decimal(data, "tick_size", "0.001")
        assert result == Decimal("0.001")

    def test_safe_decimal_none(self):
        """Test _safe_decimal with None value."""
        result = BinanceFuturesExchangeInfoAdapter._safe_decimal(None, Decimal("0.01"))
        assert result == Decimal("0.01")

    def test_safe_decimal_valid(self):
        """Test _safe_decimal with valid value."""
        result = BinanceFuturesExchangeInfoAdapter._safe_decimal("0.05", Decimal("0.01"))
        assert result == Decimal("0.05")

    def test_get_symbols_empty_cache(self):
        """Test get_symbols with empty cache and no auto_refresh."""
        adapter = BinanceFuturesExchangeInfoAdapter(config={"auto_refresh": False})
        result = adapter.get_symbols()
        assert result == []

    def test_get_max_leverage_from_brackets(self):
        """Test get_max_leverage with brackets."""
        adapter = BinanceFuturesExchangeInfoAdapter(config={"auto_refresh": False})
        adapter._leverage_brackets = {
            "BTCUSDT": [
                LeverageBracket(
                    bracket=1,
                    notional_cap=Decimal("50000"),
                    maint_margin_rate=Decimal("0.004"),
                    max_leverage=125,
                    cum_maintenance=Decimal("0"),
                ),
                LeverageBracket(
                    bracket=2,
                    notional_cap=Decimal("250000"),
                    maint_margin_rate=Decimal("0.005"),
                    max_leverage=100,
                    cum_maintenance=Decimal("50"),
                ),
            ],
        }

        # Small notional -> first bracket
        assert adapter.get_max_leverage("BTCUSDT", Decimal("10000")) == 125

        # Larger notional -> second bracket
        assert adapter.get_max_leverage("BTCUSDT", Decimal("100000")) == 100

    def test_get_margin_requirement_calculation(self):
        """Test get_margin_requirement calculation."""
        adapter = BinanceFuturesExchangeInfoAdapter(config={"auto_refresh": False})
        adapter._leverage_brackets = {
            "BTCUSDT": [
                LeverageBracket(
                    bracket=1,
                    notional_cap=Decimal("50000"),
                    maint_margin_rate=Decimal("0.004"),
                    max_leverage=125,
                    cum_maintenance=Decimal("0"),
                ),
            ],
        }

        result = adapter.get_margin_requirement("BTCUSDT", Decimal("10000"), leverage=50)

        assert result["initial"] == Decimal("200")  # 10000 / 50
        assert result["maintenance"] == Decimal("40")  # 10000 * 0.004 - 0
        assert result["notional"] == Decimal("10000")

    def test_get_perpetual_symbols(self):
        """Test get_perpetual_symbols method."""
        adapter = BinanceFuturesExchangeInfoAdapter(config={"auto_refresh": False})
        # Should work but return empty if no cache
        result = adapter.get_perpetual_symbols()
        assert isinstance(result, list)


# =============================================================================
# BinanceFuturesOrderExecutionAdapter Tests
# =============================================================================


class TestBinanceFuturesOrderExecutionAdapter:
    """Tests for BinanceFuturesOrderExecutionAdapter."""

    def test_adapter_initialization_default(self):
        """Test adapter initialization with defaults."""
        adapter = BinanceFuturesOrderExecutionAdapter()
        assert adapter._vendor == ExchangeVendor.BINANCE
        assert adapter._futures_url == "https://fapi.binance.com"
        assert adapter._hedge_mode is False

    def test_adapter_initialization_with_keys(self):
        """Test adapter initialization with API keys."""
        adapter = BinanceFuturesOrderExecutionAdapter(
            config={"api_key": "test_key", "api_secret": "test_secret"}
        )
        assert adapter._api_key == "test_key"
        assert adapter._api_secret == "test_secret"

    def test_adapter_initialization_testnet(self):
        """Test adapter initialization with testnet."""
        adapter = BinanceFuturesOrderExecutionAdapter(config={"testnet": True})
        assert adapter._futures_url == "https://testnet.binancefuture.com"

    def test_adapter_initialization_hedge_mode(self):
        """Test adapter initialization with hedge mode."""
        adapter = BinanceFuturesOrderExecutionAdapter(config={"hedge_mode": True})
        assert adapter._hedge_mode is True

    def test_market_type_property(self):
        """Test market_type property."""
        adapter = BinanceFuturesOrderExecutionAdapter()
        assert adapter.market_type == MarketType.CRYPTO_FUTURES

    def test_convert_order_type_market(self):
        """Test _convert_order_type for MARKET."""
        adapter = BinanceFuturesOrderExecutionAdapter()
        assert adapter._convert_order_type(FuturesOrderType.MARKET) == "MARKET"

    def test_convert_order_type_limit(self):
        """Test _convert_order_type for LIMIT."""
        adapter = BinanceFuturesOrderExecutionAdapter()
        assert adapter._convert_order_type(FuturesOrderType.LIMIT) == "LIMIT"

    def test_convert_order_type_stop_market(self):
        """Test _convert_order_type for STOP_MARKET."""
        adapter = BinanceFuturesOrderExecutionAdapter()
        assert adapter._convert_order_type(FuturesOrderType.STOP_MARKET) == "STOP_MARKET"

    def test_convert_order_type_take_profit_market(self):
        """Test _convert_order_type for TAKE_PROFIT_MARKET."""
        adapter = BinanceFuturesOrderExecutionAdapter()
        assert (
            adapter._convert_order_type(FuturesOrderType.TAKE_PROFIT_MARKET) == "TAKE_PROFIT_MARKET"
        )

    def test_parse_order_response_success(self, mock_order_response):
        """Test _parse_order_response for successful order."""
        adapter = BinanceFuturesOrderExecutionAdapter()
        result = adapter._parse_order_response(mock_order_response)

        assert result.success is True
        assert result.order_id == "123456789"
        assert result.client_order_id == "test_order_1"
        assert result.status == "FILLED"
        assert result.filled_qty == Decimal("0.1")
        assert result.filled_price == Decimal("50100.00")

    def test_parse_order_response_error(self):
        """Test _parse_order_response for error response."""
        adapter = BinanceFuturesOrderExecutionAdapter()
        error_response = {
            "code": -2010,
            "msg": "Order would immediately match and take.",
        }
        result = adapter._parse_order_response(error_response)

        assert result.success is False
        assert result.error_code == "-2010"

    def test_parse_order_response_invalid(self):
        """Test _parse_order_response for invalid response."""
        adapter = BinanceFuturesOrderExecutionAdapter()
        result = adapter._parse_order_response("invalid")

        assert result.success is False

    def test_parse_futures_order_response_with_fill(self, mock_order_response):
        """Test _parse_futures_order_response with filled order."""
        adapter = BinanceFuturesOrderExecutionAdapter()
        result = adapter._parse_futures_order_response(mock_order_response)

        assert isinstance(result, FuturesOrderResult)
        assert result.success is True
        assert result.position_side == "BOTH"
        assert result.futures_fill is not None
        assert result.futures_fill.filled_qty == Decimal("0.1")

    @patch.object(BinanceFuturesOrderExecutionAdapter, "_request")
    def test_set_leverage(self, mock_request):
        """Test set_leverage method."""
        mock_request.return_value = {"leverage": 10, "symbol": "BTCUSDT"}

        adapter = BinanceFuturesOrderExecutionAdapter(
            config={"api_key": "test", "api_secret": "test"}
        )
        result = adapter.set_leverage("BTCUSDT", 10)

        assert result is True
        mock_request.assert_called_once()

    @patch.object(BinanceFuturesOrderExecutionAdapter, "_request")
    def test_set_margin_type(self, mock_request):
        """Test set_margin_type method."""
        mock_request.return_value = {"code": 200, "msg": "success"}

        adapter = BinanceFuturesOrderExecutionAdapter(
            config={"api_key": "test", "api_secret": "test"}
        )
        result = adapter.set_margin_type("BTCUSDT", "CROSSED")

        assert result is True

    @patch.object(BinanceFuturesOrderExecutionAdapter, "_request")
    def test_get_futures_positions(self, mock_request, mock_position_response):
        """Test get_futures_positions method."""
        mock_request.return_value = mock_position_response

        adapter = BinanceFuturesOrderExecutionAdapter(
            config={"api_key": "test", "api_secret": "test"}
        )
        result = adapter.get_futures_positions()

        assert "BTCUSDT" in result
        position = result["BTCUSDT"]
        assert isinstance(position, FuturesPosition)
        assert position.qty == Decimal("0.5")
        assert position.leverage == 20
        assert position.margin_mode == MarginMode.CROSS

    @patch.object(BinanceFuturesOrderExecutionAdapter, "_request")
    def test_get_futures_account_state(self, mock_request, mock_account_response):
        """Test get_futures_account_state method."""
        mock_request.return_value = mock_account_response

        adapter = BinanceFuturesOrderExecutionAdapter(
            config={"api_key": "test", "api_secret": "test"}
        )
        result = adapter.get_futures_account_state()

        assert result is not None
        assert isinstance(result, FuturesAccountState)
        assert result.total_wallet_balance == Decimal("10000.00")
        assert result.available_balance == Decimal("8000.00")

    @patch.object(BinanceFuturesOrderExecutionAdapter, "_request")
    def test_get_account_info(self, mock_request, mock_account_response):
        """Test get_account_info method."""
        mock_request.return_value = mock_account_response

        adapter = BinanceFuturesOrderExecutionAdapter(
            config={"api_key": "test", "api_secret": "test"}
        )
        result = adapter.get_account_info()

        assert result.account_type == "futures"
        assert result.buying_power == Decimal("8000.00")
        assert result.cash_balance == Decimal("10000.00")


# =============================================================================
# Registry Integration Tests
# =============================================================================


class TestRegistryIntegration:
    """Tests for registry integration with futures adapters."""

    def test_create_futures_market_data_adapter_binance(self):
        """Test creating futures market data adapter via registry."""
        adapter = create_futures_market_data_adapter("binance_futures")
        assert isinstance(adapter, BinanceFuturesMarketDataAdapter)

    def test_create_futures_market_data_adapter_binance_vendor(self):
        """Test creating futures adapter via standard binance vendor."""
        adapter = create_futures_market_data_adapter("binance")
        assert isinstance(adapter, BinanceFuturesMarketDataAdapter)

    def test_create_futures_exchange_info_adapter(self):
        """Test creating futures exchange info adapter via registry."""
        adapter = create_futures_exchange_info_adapter("binance_futures")
        assert isinstance(adapter, BinanceFuturesExchangeInfoAdapter)

    def test_create_futures_order_execution_adapter(self):
        """Test creating futures order execution adapter via registry."""
        adapter = create_futures_order_execution_adapter(
            "binance_futures",
            config={"api_key": "test", "api_secret": "test"},
        )
        assert isinstance(adapter, BinanceFuturesOrderExecutionAdapter)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_market_data_adapter_none_response(self):
        """Test market data adapter with None response."""
        adapter = BinanceFuturesMarketDataAdapter()
        with patch.object(adapter, "_get_client") as mock:
            mock.return_value._session_get.return_value = None
            result = adapter.get_mark_price("BTCUSDT")
            assert result is None

    def test_market_data_adapter_empty_list_response(self):
        """Test market data adapter with empty list response."""
        adapter = BinanceFuturesMarketDataAdapter()
        with patch.object(adapter, "_get_client") as mock:
            mock.return_value._session_get.return_value = []
            results = adapter.get_mark_prices()
            assert results == []

    def test_exchange_info_adapter_cache_not_found(self):
        """Test exchange info adapter when cache file not found."""
        adapter = BinanceFuturesExchangeInfoAdapter(
            config={"filters_cache_path": "/nonexistent/path.json", "auto_refresh": False}
        )
        assert len(adapter._symbols_cache) == 0

    def test_order_execution_adapter_cancel_no_symbol(self):
        """Test order cancellation without symbol."""
        adapter = BinanceFuturesOrderExecutionAdapter(
            config={"api_key": "test", "api_secret": "test"}
        )
        result = adapter.cancel_order(order_id="123")
        assert result is False

    def test_order_execution_adapter_cancel_no_ids(self):
        """Test order cancellation without any ID."""
        adapter = BinanceFuturesOrderExecutionAdapter(
            config={"api_key": "test", "api_secret": "test"}
        )
        result = adapter.cancel_order(symbol="BTCUSDT")
        assert result is False

    @patch.object(BinanceFuturesOrderExecutionAdapter, "get_futures_positions")
    def test_close_position_no_position(self, mock_get_positions):
        """Test close_position when no position exists."""
        mock_get_positions.return_value = {}

        adapter = BinanceFuturesOrderExecutionAdapter(
            config={"api_key": "test", "api_secret": "test"}
        )
        result = adapter.close_position("BTCUSDT")

        assert result.success is True
        assert result.status == "NO_POSITION"

    def test_parse_exec_report_various_statuses(self):
        """Test _parse_exec_report with various statuses."""
        adapter = BinanceFuturesOrderExecutionAdapter()

        # Test FILLED
        filled = adapter._parse_exec_report(
            {
                "status": "FILLED",
                "side": "BUY",
                "symbol": "BTCUSDT",
                "orderId": "123",
            }
        )
        assert filled.exec_status.value == "FILLED"

        # Test CANCELED
        canceled = adapter._parse_exec_report(
            {
                "status": "CANCELED",
                "side": "SELL",
                "symbol": "BTCUSDT",
                "orderId": "456",
            }
        )
        assert canceled.exec_status.value == "CANCELED"

        # Test REJECTED
        rejected = adapter._parse_exec_report(
            {
                "status": "REJECTED",
                "side": "BUY",
                "symbol": "BTCUSDT",
                "orderId": "789",
            }
        )
        assert rejected.exec_status.value == "REJECTED"


# =============================================================================
# Contract Spec Parsing Tests
# =============================================================================


class TestContractSpecParsing:
    """Tests for contract specification parsing."""

    def test_parse_perpetual_contract(self):
        """Test parsing perpetual contract from exchange info."""
        adapter = BinanceFuturesExchangeInfoAdapter(config={"auto_refresh": False})
        symbol_data = {
            "symbol": "BTCUSDT",
            "baseAsset": "BTC",
            "quoteAsset": "USDT",
            "marginAsset": "USDT",
            "contractType": "PERPETUAL",
            "filters": [
                {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                {
                    "filterType": "LOT_SIZE",
                    "stepSize": "0.001",
                    "minQty": "0.001",
                    "maxQty": "1000",
                },
            ],
        }

        spec = adapter._parse_contract_spec(symbol_data)

        assert spec is not None
        assert spec.symbol == "BTCUSDT"
        assert spec.futures_type == FuturesType.CRYPTO_PERPETUAL
        assert spec.contract_type == ContractType.PERPETUAL
        assert spec.tick_size == Decimal("0.01")

    def test_parse_quarterly_contract(self):
        """Test parsing quarterly contract from exchange info."""
        adapter = BinanceFuturesExchangeInfoAdapter(config={"auto_refresh": False})
        symbol_data = {
            "symbol": "BTCUSD_240329",
            "baseAsset": "BTC",
            "quoteAsset": "USD",
            "marginAsset": "BTC",
            "contractType": "CURRENT_QUARTER",
            "deliveryDate": 1711670400000,
            "filters": [],
        }

        spec = adapter._parse_contract_spec(symbol_data)

        assert spec is not None
        assert spec.futures_type == FuturesType.CRYPTO_QUARTERLY
        assert spec.contract_type == ContractType.CURRENT_QUARTER
        assert spec.delivery_date is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
