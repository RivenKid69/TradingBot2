# -*- coding: utf-8 -*-
"""
Tests for LOB Data Adapters (Stage 8).

Tests cover:
1. LOBSTERAdapter - LOBSTER format parsing
2. ITCHAdapter - ITCH format parsing
3. BinanceL2Adapter - Binance depth data
4. AlpacaL2Adapter - Alpaca market data
5. Factory functions
6. Data conversions and statistics

Total: 25+ tests
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest
pytest.importorskip("sortedcontainers")

from lob.data_adapters import (
    AdapterStats,
    AlpacaL2Adapter,
    BaseLOBAdapter,
    BinanceL2Adapter,
    DataSourceType,
    DepthLevel,
    DepthSnapshot,
    ITCHAdapter,
    LOBSTERAdapter,
    LOBUpdate,
    create_lob_adapter,
    load_orderbook_from_file,
)
from lob.data_structures import OrderBook, Side


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def lobster_message_file(tmp_path: Path) -> Path:
    """Create a temporary LOBSTER message file."""
    content = """34200.123456789,1,12345,100,100.50,1
34200.234567890,1,12346,200,100.55,-1
34200.345678901,4,12345,50,100.50,1
34200.456789012,3,12346,200,100.55,-1
34200.567890123,1,12347,150,100.45,1
"""
    file_path = tmp_path / "messages.csv"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def lobster_orderbook_file(tmp_path: Path) -> Path:
    """Create a temporary LOBSTER orderbook file."""
    # Format: ask_price_1, ask_size_1, bid_price_1, bid_size_1, ...
    content = "1005500,200,1005000,100,1006000,150,1004500,120"
    file_path = tmp_path / "orderbook.csv"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def binance_depth_data() -> Dict[str, Any]:
    """Create sample Binance depth data."""
    return {
        "lastUpdateId": 123456,
        "bids": [
            ["100.00", "10.5"],
            ["99.95", "20.0"],
            ["99.90", "15.0"],
        ],
        "asks": [
            ["100.05", "8.0"],
            ["100.10", "12.0"],
            ["100.15", "25.0"],
        ],
    }


@pytest.fixture
def alpaca_quote_data() -> Dict[str, Any]:
    """Create sample Alpaca quote data."""
    return {
        "t": "2024-01-15T10:30:00Z",
        "bp": 150.25,
        "bs": 100,
        "ap": 150.30,
        "as": 200,
    }


# ==============================================================================
# Test LOBSTERAdapter
# ==============================================================================


class TestLOBSTERAdapter:
    """Tests for LOBSTERAdapter."""

    def test_init(self) -> None:
        """Test adapter initialization."""
        adapter = LOBSTERAdapter(symbol="AAPL")
        assert adapter.source_type == DataSourceType.LOBSTER
        assert adapter.symbol == "AAPL"
        assert adapter.stats.messages_processed == 0

    def test_parse_message_file(self, lobster_message_file: Path) -> None:
        """Test parsing LOBSTER message file."""
        adapter = LOBSTERAdapter(
            symbol="TEST",
            config={"price_multiplier": 0.01},  # Price in cents
        )

        messages = list(adapter.parse_message_file(lobster_message_file))
        assert len(messages) == 5

        # First message: ADD BUY
        msg0 = messages[0]
        assert msg0.event_type.value == 1  # ADD
        assert msg0.order_id == 12345
        assert msg0.size == 100
        assert msg0.direction == 1  # BUY
        assert msg0.is_add

        # Second message: ADD SELL
        msg1 = messages[1]
        assert msg1.direction == -1  # SELL

        # Third message: EXECUTE
        msg2 = messages[2]
        assert msg2.event_type.value == 4  # EXECUTE
        assert msg2.is_execute

        # Fourth message: DELETE
        msg3 = messages[3]
        assert msg3.event_type.value == 3  # DELETE
        assert msg3.is_delete

    def test_stream_updates(self, lobster_message_file: Path) -> None:
        """Test streaming updates from file."""
        adapter = LOBSTERAdapter(config={"price_multiplier": 0.01})

        updates = list(adapter.stream_updates(lobster_message_file))
        assert len(updates) == 5

        # Check update types
        assert updates[0].update_type == "ADD"
        assert updates[2].update_type == "EXECUTE"
        assert updates[3].update_type == "DELETE"

        # Check stats
        assert adapter.stats.messages_processed == 5
        assert adapter.stats.orders_added == 3
        assert adapter.stats.orders_executed == 1
        assert adapter.stats.orders_cancelled == 1

    def test_parse_orderbook_file(self, lobster_orderbook_file: Path) -> None:
        """Test parsing LOBSTER orderbook file."""
        adapter = LOBSTERAdapter(config={"price_multiplier": 0.0001})

        snapshot = adapter.parse_orderbook_file(lobster_orderbook_file, n_levels=2)

        assert len(snapshot.asks) == 2
        assert len(snapshot.bids) == 2

        # Best ask
        assert snapshot.asks[0].price == pytest.approx(100.55, rel=0.01)
        assert snapshot.asks[0].qty == 200

        # Best bid
        assert snapshot.bids[0].price == pytest.approx(100.50, rel=0.01)
        assert snapshot.bids[0].qty == 100

    def test_build_orderbook_from_messages(self, lobster_message_file: Path) -> None:
        """Test building OrderBook by replaying messages."""
        adapter = LOBSTERAdapter(config={"price_multiplier": 0.01})

        book = adapter.build_orderbook_from_messages(lobster_message_file)

        # Should have orders remaining after processing
        assert isinstance(book, OrderBook)

    def test_load_snapshot(self, lobster_orderbook_file: Path) -> None:
        """Test load_snapshot interface."""
        adapter = LOBSTERAdapter(config={"price_multiplier": 0.0001})

        snapshot = adapter.load_snapshot(lobster_orderbook_file)

        assert snapshot.bids
        assert snapshot.asks

    def test_max_messages_limit(self, lobster_message_file: Path) -> None:
        """Test limiting number of messages."""
        adapter = LOBSTERAdapter(config={"price_multiplier": 0.01})

        updates = list(adapter.stream_updates(lobster_message_file, max_messages=2))
        assert len(updates) == 2


class TestITCHAdapter:
    """Tests for ITCHAdapter."""

    def test_init(self) -> None:
        """Test adapter initialization."""
        adapter = ITCHAdapter(symbol="AAPL")
        assert adapter.source_type == DataSourceType.ITCH
        assert adapter.symbol == "AAPL"

    def test_load_snapshot_returns_empty(self) -> None:
        """Test that load_snapshot returns empty for ITCH (no snapshot format)."""
        adapter = ITCHAdapter(symbol="AAPL")
        snapshot = adapter.load_snapshot("dummy_path")

        assert snapshot.bids == []
        assert snapshot.asks == []


# ==============================================================================
# Test BinanceL2Adapter
# ==============================================================================


class TestBinanceL2Adapter:
    """Tests for BinanceL2Adapter."""

    def test_init(self) -> None:
        """Test adapter initialization."""
        adapter = BinanceL2Adapter(symbol="BTCUSDT")
        assert adapter.source_type == DataSourceType.BINANCE
        assert adapter.symbol == "BTCUSDT"

    def test_load_snapshot(self, binance_depth_data: Dict[str, Any]) -> None:
        """Test loading snapshot from Binance depth data."""
        adapter = BinanceL2Adapter(symbol="BTCUSDT")

        snapshot = adapter.load_snapshot(binance_depth_data)

        assert len(snapshot.bids) == 3
        assert len(snapshot.asks) == 3

        # Best bid
        assert snapshot.bids[0].price == 100.00
        assert snapshot.bids[0].qty == 10.5

        # Best ask
        assert snapshot.asks[0].price == 100.05
        assert snapshot.asks[0].qty == 8.0

    def test_load_snapshot_from_file(
        self, tmp_path: Path, binance_depth_data: Dict[str, Any]
    ) -> None:
        """Test loading snapshot from JSON file."""
        file_path = tmp_path / "depth.json"
        file_path.write_text(json.dumps(binance_depth_data))

        adapter = BinanceL2Adapter(symbol="BTCUSDT")
        snapshot = adapter.load_snapshot(str(file_path))

        assert len(snapshot.bids) == 3

    def test_depth_to_orderbook(self, binance_depth_data: Dict[str, Any]) -> None:
        """Test converting depth snapshot to OrderBook."""
        adapter = BinanceL2Adapter(symbol="BTCUSDT")

        book = adapter.depth_to_orderbook(binance_depth_data)

        assert isinstance(book, OrderBook)
        assert book.best_bid is not None
        assert book.best_ask is not None
        assert book.best_bid == pytest.approx(100.00, rel=0.01)
        assert book.best_ask == pytest.approx(100.05, rel=0.01)

    def test_stream_updates(self, binance_depth_data: Dict[str, Any]) -> None:
        """Test streaming depth updates."""
        adapter = BinanceL2Adapter(symbol="BTCUSDT")

        # Simulate depth update stream
        update_stream = [
            {
                "E": 1705315800000,  # Event time in ms
                "b": [["100.00", "15.0"]],  # Bid update
                "a": [["100.05", "0"]],  # Ask removed (qty=0)
            }
        ]

        updates = list(adapter.stream_updates(iter(update_stream)))

        assert len(updates) == 2  # One bid ADD, one ask DELETE

        # Check bid update
        bid_update = [u for u in updates if u.side == Side.BUY][0]
        assert bid_update.update_type == "ADD"
        assert bid_update.price == 100.00
        assert bid_update.qty == 15.0

        # Check ask delete
        ask_update = [u for u in updates if u.side == Side.SELL][0]
        assert ask_update.update_type == "DELETE"
        assert ask_update.qty == 0


# ==============================================================================
# Test AlpacaL2Adapter
# ==============================================================================


class TestAlpacaL2Adapter:
    """Tests for AlpacaL2Adapter."""

    def test_init(self) -> None:
        """Test adapter initialization."""
        adapter = AlpacaL2Adapter(
            symbol="AAPL",
            config={"feed": "iex"},
        )
        assert adapter.source_type == DataSourceType.ALPACA
        assert adapter.symbol == "AAPL"

    def test_load_snapshot(self, alpaca_quote_data: Dict[str, Any]) -> None:
        """Test loading snapshot from Alpaca quote data."""
        adapter = AlpacaL2Adapter(symbol="AAPL")

        snapshot = adapter.load_snapshot(alpaca_quote_data)

        # Alpaca provides only top-of-book
        assert len(snapshot.bids) == 1
        assert len(snapshot.asks) == 1

        assert snapshot.bids[0].price == 150.25
        assert snapshot.bids[0].qty == 100
        assert snapshot.asks[0].price == 150.30
        assert snapshot.asks[0].qty == 200

    def test_stream_updates(self) -> None:
        """Test streaming quote updates."""
        adapter = AlpacaL2Adapter(symbol="AAPL")

        # Simulate quote stream
        quote_stream = [
            {
                "t": "2024-01-15T10:30:00Z",
                "bp": 150.25,
                "bs": 100,
                "ap": 150.30,
                "as": 200,
            },
            {
                "t": "2024-01-15T10:30:01Z",
                "bp": 150.26,  # Bid moved up
                "bs": 150,
                "ap": 150.30,
                "as": 200,
            },
        ]

        updates = list(adapter.stream_updates(iter(quote_stream)))

        # First quote: 1 bid ADD, 1 ask ADD
        # Second quote: 1 bid DELETE + 1 bid ADD (bid changed)
        assert len(updates) >= 2

    def test_quotes_to_depth_snapshot(self) -> None:
        """Test converting quotes to DepthSnapshot."""
        adapter = AlpacaL2Adapter(symbol="AAPL")

        quotes = [
            {"bp": 150.25, "bs": 100, "ap": 150.30, "as": 200},
        ]

        snapshot = adapter.quotes_to_depth_snapshot(quotes)

        assert isinstance(snapshot, DepthSnapshot)
        assert len(snapshot.bids) == 1
        assert len(snapshot.asks) == 1


# ==============================================================================
# Test Data Structures
# ==============================================================================


class TestDataStructures:
    """Tests for adapter data structures."""

    def test_depth_level(self) -> None:
        """Test DepthLevel dataclass."""
        level = DepthLevel(price=100.0, qty=50.0, order_count=3)
        assert level.price == 100.0
        assert level.qty == 50.0
        assert level.order_count == 3

    def test_depth_snapshot_properties(self) -> None:
        """Test DepthSnapshot calculated properties."""
        snapshot = DepthSnapshot(
            timestamp_ns=1000000000,
            symbol="TEST",
            bids=[DepthLevel(100.0, 50.0), DepthLevel(99.0, 30.0)],
            asks=[DepthLevel(101.0, 40.0), DepthLevel(102.0, 20.0)],
        )

        assert snapshot.best_bid == 100.0
        assert snapshot.best_ask == 101.0
        assert snapshot.mid_price == 100.5
        assert snapshot.spread_bps == pytest.approx(100.0, rel=0.1)  # 1%

    def test_depth_snapshot_empty(self) -> None:
        """Test DepthSnapshot with no depth."""
        snapshot = DepthSnapshot(
            timestamp_ns=1000000000,
            symbol="TEST",
        )

        assert snapshot.best_bid is None
        assert snapshot.best_ask is None
        assert snapshot.mid_price is None
        assert snapshot.spread_bps is None

    def test_lob_update(self) -> None:
        """Test LOBUpdate dataclass."""
        update = LOBUpdate(
            timestamp_ns=1000000000,
            update_type="ADD",
            order_id="order_123",
            price=100.0,
            qty=50.0,
            side=Side.BUY,
            source=DataSourceType.LOBSTER,
        )

        assert update.update_type == "ADD"
        assert update.side == Side.BUY
        assert update.source == DataSourceType.LOBSTER

    def test_adapter_stats(self) -> None:
        """Test AdapterStats calculations."""
        stats = AdapterStats(
            messages_processed=100,
            messages_failed=5,
            orders_added=50,
            orders_cancelled=20,
            orders_executed=30,
            total_volume=10000.0,
            start_time_ns=0,
            end_time_ns=60_000_000_000,  # 60 seconds
        )

        assert stats.duration_sec == 60.0
        # error_rate = failed / (processed + failed) = 5 / 105
        assert stats.error_rate == pytest.approx(5 / 105, rel=0.01)


# ==============================================================================
# Test Factory Functions
# ==============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_lob_adapter_lobster(self) -> None:
        """Test creating LOBSTER adapter via factory."""
        adapter = create_lob_adapter("lobster", symbol="AAPL")
        assert isinstance(adapter, LOBSTERAdapter)
        assert adapter.symbol == "AAPL"

    def test_create_lob_adapter_itch(self) -> None:
        """Test creating ITCH adapter via factory."""
        adapter = create_lob_adapter(DataSourceType.ITCH, symbol="MSFT")
        assert isinstance(adapter, ITCHAdapter)
        assert adapter.symbol == "MSFT"

    def test_create_lob_adapter_binance(self) -> None:
        """Test creating Binance adapter via factory."""
        adapter = create_lob_adapter("binance", symbol="BTCUSDT")
        assert isinstance(adapter, BinanceL2Adapter)

    def test_create_lob_adapter_alpaca(self) -> None:
        """Test creating Alpaca adapter via factory."""
        adapter = create_lob_adapter("alpaca", symbol="GOOGL")
        assert isinstance(adapter, AlpacaL2Adapter)

    def test_create_lob_adapter_unknown(self) -> None:
        """Test creating adapter with unknown type."""
        with pytest.raises(ValueError, match="is not a valid DataSourceType"):
            create_lob_adapter("unknown_format")

    def test_create_lob_adapter_with_config(self) -> None:
        """Test creating adapter with custom config."""
        config = {"price_multiplier": 0.001, "delimiter": ";"}
        adapter = create_lob_adapter("lobster", symbol="TEST", config=config)

        assert isinstance(adapter, LOBSTERAdapter)


# ==============================================================================
# Test Integration
# ==============================================================================


class TestIntegration:
    """Integration tests for data adapters."""

    def test_lobster_full_workflow(self, lobster_message_file: Path) -> None:
        """Test full LOBSTER workflow: parse → stream → build book."""
        adapter = LOBSTERAdapter(
            symbol="TEST",
            config={"price_multiplier": 0.01},
        )

        # Stream updates
        updates = list(adapter.stream_updates(lobster_message_file))
        assert len(updates) > 0

        # Build order book
        book = adapter.build_orderbook_from_messages(lobster_message_file)
        assert isinstance(book, OrderBook)

        # Check stats
        assert adapter.stats.messages_processed > 0
        assert adapter.stats.duration_sec > 0

    def test_binance_depth_workflow(self, binance_depth_data: Dict[str, Any]) -> None:
        """Test Binance workflow: load → convert → book."""
        adapter = BinanceL2Adapter(symbol="BTCUSDT")

        # Load snapshot
        snapshot = adapter.load_snapshot(binance_depth_data)
        assert snapshot.bids

        # Convert to OrderBook
        book = adapter.depth_to_orderbook(binance_depth_data)
        assert book.best_bid is not None
        assert book.best_ask is not None

    def test_reset_stats(self, lobster_message_file: Path) -> None:
        """Test resetting adapter statistics."""
        adapter = LOBSTERAdapter(config={"price_multiplier": 0.01})

        # Process some messages
        list(adapter.stream_updates(lobster_message_file))
        assert adapter.stats.messages_processed > 0

        # Reset stats
        adapter.reset_stats()
        assert adapter.stats.messages_processed == 0
        assert adapter.stats.orders_added == 0


# ==============================================================================
# Test Edge Cases
# ==============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_depth_data(self) -> None:
        """Test handling empty depth data."""
        adapter = BinanceL2Adapter(symbol="TEST")

        empty_data = {"lastUpdateId": 0, "bids": [], "asks": []}
        snapshot = adapter.load_snapshot(empty_data)

        assert len(snapshot.bids) == 0
        assert len(snapshot.asks) == 0

    def test_malformed_lobster_lines(self, tmp_path: Path) -> None:
        """Test handling malformed LOBSTER data."""
        content = """34200.123456789,1,12345,100,100.50,1
invalid_line
34200.234567890,1,12346,200,100.55,-1
"""
        file_path = tmp_path / "bad_messages.csv"
        file_path.write_text(content)

        adapter = LOBSTERAdapter(config={"price_multiplier": 0.01})
        messages = list(adapter.parse_message_file(file_path))

        # Should skip invalid line
        assert len(messages) == 2

    def test_zero_quantity_binance(self) -> None:
        """Test Binance adapter with zero quantity (delete signal)."""
        adapter = BinanceL2Adapter(symbol="TEST")

        update_stream = [
            {
                "E": 1705315800000,
                "b": [["100.00", "0"]],  # Qty = 0 means delete
                "a": [],
            }
        ]

        updates = list(adapter.stream_updates(iter(update_stream)))
        assert len(updates) == 1
        assert updates[0].update_type == "DELETE"

    def test_alpaca_missing_quote_fields(self) -> None:
        """Test Alpaca adapter with missing quote fields."""
        adapter = AlpacaL2Adapter(symbol="TEST")

        quote_data = {"t": "2024-01-15T10:30:00Z"}  # No bid/ask
        snapshot = adapter.load_snapshot(quote_data)

        assert len(snapshot.bids) == 0
        assert len(snapshot.asks) == 0


# ==============================================================================
# Test Bug Fixes (Stage 8 Review)
# ==============================================================================


class TestBugFixes:
    """Tests for bug fixes from Stage 8 code review."""

    def test_build_orderbook_with_initial_snapshot(
        self, lobster_message_file: Path, lobster_orderbook_file: Path
    ) -> None:
        """Test build_orderbook_from_messages with orderbook_path (Bug #1 fix)."""
        adapter = LOBSTERAdapter(
            symbol="TEST",
            config={"price_multiplier": 0.0001},
        )

        # This should work now after the fix (was using bid_levels instead of bids)
        book = adapter.build_orderbook_from_messages(
            message_path=lobster_message_file,
            orderbook_path=lobster_orderbook_file,
        )

        assert isinstance(book, OrderBook)
        # Should have orders from both initial snapshot and messages
        assert book.best_bid is not None or book.best_ask is not None

    def test_load_orderbook_from_file_binance(
        self, tmp_path: Path, binance_depth_data: Dict[str, Any]
    ) -> None:
        """Test load_orderbook_from_file with Binance data (Bug #1 fix)."""
        # Create depth file
        file_path = tmp_path / "depth.json"
        file_path.write_text(json.dumps(binance_depth_data))

        # This should work now after the fix (was using bid_levels/ask_levels)
        book = load_orderbook_from_file(str(file_path), source_type="binance", symbol="TEST")

        assert isinstance(book, OrderBook)
        assert book.best_bid is not None
        assert book.best_ask is not None
        # Verify bids and asks are loaded
        assert book.best_bid == pytest.approx(100.00, rel=0.01)
        assert book.best_ask == pytest.approx(100.05, rel=0.01)

    def test_depth_level_is_dataclass_not_dict(self) -> None:
        """Verify DepthLevel is a dataclass and accessed via attributes."""
        level = DepthLevel(price=100.0, qty=50.0, order_count=3)

        # Access via attributes, not dict keys
        assert level.price == 100.0
        assert level.qty == 50.0
        assert level.order_count == 3

        # Verify it's NOT a dict
        assert not isinstance(level, dict)
        with pytest.raises(TypeError):
            _ = level["price"]  # type: ignore

    def test_depth_snapshot_has_bids_asks_not_levels(self) -> None:
        """Verify DepthSnapshot uses 'bids'/'asks' not 'bid_levels'/'ask_levels'."""
        snapshot = DepthSnapshot(
            timestamp_ns=1000000000,
            symbol="TEST",
            bids=[DepthLevel(100.0, 50.0)],
            asks=[DepthLevel(101.0, 40.0)],
        )

        # Has bids/asks
        assert hasattr(snapshot, "bids")
        assert hasattr(snapshot, "asks")
        assert len(snapshot.bids) == 1
        assert len(snapshot.asks) == 1

        # Does NOT have bid_levels/ask_levels
        assert not hasattr(snapshot, "bid_levels")
        assert not hasattr(snapshot, "ask_levels")


# ==============================================================================
# Test AlpacaL2Adapter Historical Data Methods (Task 4 - L3 LOB for Stocks)
# ==============================================================================


class TestAlpacaL2AdapterHistorical:
    """Tests for AlpacaL2Adapter historical data fetching methods."""

    def test_fetch_historical_quotes_no_credentials(self) -> None:
        """Test fetch_historical_quotes returns empty without credentials."""
        adapter = AlpacaL2Adapter(
            symbol="AAPL",
            config={"api_key": "", "api_secret": ""},
        )

        quotes = adapter.fetch_historical_quotes("AAPL", "2025-01-01", "2025-01-02")
        assert quotes == []

    def test_fetch_historical_trades_no_credentials(self) -> None:
        """Test fetch_historical_trades returns empty without credentials."""
        adapter = AlpacaL2Adapter(
            symbol="AAPL",
            config={"api_key": "", "api_secret": ""},
        )

        trades = adapter.fetch_historical_trades("AAPL", "2025-01-01", "2025-01-02")
        assert trades == []

    def test_fetch_historical_quotes_no_symbol(self) -> None:
        """Test fetch_historical_quotes returns empty with no symbol."""
        adapter = AlpacaL2Adapter()  # No symbol specified

        quotes = adapter.fetch_historical_quotes()
        assert quotes == []

    def test_fetch_historical_trades_no_symbol(self) -> None:
        """Test fetch_historical_trades returns empty with no symbol."""
        adapter = AlpacaL2Adapter()

        trades = adapter.fetch_historical_trades()
        assert trades == []

    def test_fetch_historical_quotes_caching(self) -> None:
        """Test that quotes are cached after fetching."""
        adapter = AlpacaL2Adapter(
            symbol="AAPL",
            config={"api_key": "", "api_secret": ""},
        )

        # Manually populate cache
        cache_key = "AAPL_2025-01-01_2025-01-02"
        cached_quotes = [{"t": "2025-01-01T10:00:00Z", "bp": 150.0, "bs": 100, "ap": 150.05, "as": 200}]
        adapter._quotes_cache[cache_key] = cached_quotes

        # Should return cached value
        quotes = adapter.fetch_historical_quotes("AAPL", "2025-01-01", "2025-01-02")
        assert quotes == cached_quotes

    def test_fetch_historical_trades_caching(self) -> None:
        """Test that trades are cached after fetching."""
        adapter = AlpacaL2Adapter(
            symbol="AAPL",
            config={"api_key": "", "api_secret": ""},
        )

        # Manually populate cache
        cache_key = "AAPL_2025-01-01_2025-01-02_trades"
        cached_trades = [{"t": "2025-01-01T10:00:00Z", "p": 150.0, "s": 100}]
        adapter._trades_cache[cache_key] = cached_trades

        # Should return cached value
        trades = adapter.fetch_historical_trades("AAPL", "2025-01-01", "2025-01-02")
        assert trades == cached_trades

    def test_compute_calibration_observations_no_symbol(self) -> None:
        """Test compute_calibration_observations returns error with no symbol."""
        adapter = AlpacaL2Adapter()

        result = adapter.compute_calibration_observations()
        assert "error" in result
        assert result["error"] == "No symbol specified"

    def test_compute_calibration_observations_insufficient_data(self) -> None:
        """Test compute_calibration_observations handles insufficient data."""
        adapter = AlpacaL2Adapter(
            symbol="AAPL",
            config={"api_key": "", "api_secret": ""},
        )

        # Empty caches = no data
        result = adapter.compute_calibration_observations("AAPL", "2025-01-01", "2025-01-02")
        assert "error" in result
        assert result["error"] == "Insufficient data"
        assert result["n_quotes"] == 0
        assert result["n_trades"] == 0

    def test_compute_calibration_observations_with_data(self) -> None:
        """Test compute_calibration_observations with mocked data."""
        adapter = AlpacaL2Adapter(
            symbol="AAPL",
            config={"api_key": "", "api_secret": ""},
        )

        # Pre-populate caches with test data
        quotes_key = "AAPL_2025-01-01_2025-01-02"
        trades_key = "AAPL_2025-01-01_2025-01-02_trades"

        adapter._quotes_cache[quotes_key] = [
            {"t": "2025-01-01T10:00:00Z", "bp": 150.00, "bs": 100, "ap": 150.05, "as": 200},
            {"t": "2025-01-01T10:00:01Z", "bp": 150.01, "bs": 110, "ap": 150.06, "as": 180},
            {"t": "2025-01-01T10:00:02Z", "bp": 150.02, "bs": 120, "ap": 150.07, "as": 190},
        ]

        adapter._trades_cache[trades_key] = [
            {"t": "2025-01-01T10:00:00.500Z", "p": 150.05, "s": 50},
            {"t": "2025-01-01T10:00:01.500Z", "p": 150.01, "s": 75},
        ]

        result = adapter.compute_calibration_observations("AAPL", "2025-01-01", "2025-01-02")

        assert "error" not in result
        assert result["symbol"] == "AAPL"
        assert result["n_quotes"] == 3
        assert result["n_trades"] == 2
        assert result["avg_spread_bps"] is not None
        assert result["avg_spread_bps"] > 0
        assert "trade_observations" in result

    def test_to_calibration_pipeline_data_no_symbol(self) -> None:
        """Test to_calibration_pipeline_data returns error with no symbol."""
        adapter = AlpacaL2Adapter()

        result = adapter.to_calibration_pipeline_data()
        assert "error" in result

    def test_to_calibration_pipeline_data_with_data(self) -> None:
        """Test to_calibration_pipeline_data formats data correctly."""
        adapter = AlpacaL2Adapter(
            symbol="AAPL",
            config={"api_key": "", "api_secret": ""},
        )

        # Pre-populate caches
        quotes_key = "AAPL_None_None"
        trades_key = "AAPL_None_None_trades"

        adapter._quotes_cache[quotes_key] = [
            {"t": "2025-01-01T10:00:00Z", "bp": 150.00, "bs": 100, "ap": 150.10, "as": 200},
            {"t": "2025-01-01T10:00:02Z", "bp": 150.02, "bs": 120, "ap": 150.12, "as": 190},
        ]

        adapter._trades_cache[trades_key] = [
            {"t": "2025-01-01T10:00:01Z", "p": 150.10, "s": 50},
        ]

        result = adapter.to_calibration_pipeline_data("AAPL")

        assert "symbol" in result
        assert result["symbol"] == "AAPL"
        assert "trades" in result
        assert "market_params" in result
        assert "avg_adv" in result["market_params"]
        assert "avg_volatility" in result["market_params"]


class TestAlpacaL2AdapterMocked:
    """Tests for AlpacaL2Adapter with mocked API responses."""

    def test_fetch_historical_quotes_mocked(self) -> None:
        """Test fetch_historical_quotes with mocked API."""
        from unittest.mock import patch, MagicMock

        adapter = AlpacaL2Adapter(
            symbol="AAPL",
            config={"api_key": "test_key", "api_secret": "test_secret"},
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "quotes": {
                "AAPL": [
                    {"t": "2025-01-01T10:00:00Z", "bp": 150.0, "bs": 100, "ap": 150.05, "as": 200},
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            quotes = adapter.fetch_historical_quotes("AAPL", "2025-01-01", "2025-01-02")

        assert len(quotes) == 1
        assert quotes[0]["bp"] == 150.0
        mock_get.assert_called_once()

    def test_fetch_historical_trades_mocked(self) -> None:
        """Test fetch_historical_trades with mocked API."""
        from unittest.mock import patch, MagicMock

        adapter = AlpacaL2Adapter(
            symbol="AAPL",
            config={"api_key": "test_key", "api_secret": "test_secret"},
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "trades": {
                "AAPL": [
                    {"t": "2025-01-01T10:00:00Z", "p": 150.02, "s": 50, "c": []},
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            trades = adapter.fetch_historical_trades("AAPL", "2025-01-01", "2025-01-02")

        assert len(trades) == 1
        assert trades[0]["p"] == 150.02
        mock_get.assert_called_once()

    def test_fetch_historical_quotes_api_error(self) -> None:
        """Test fetch_historical_quotes handles API errors gracefully."""
        from unittest.mock import patch

        adapter = AlpacaL2Adapter(
            symbol="AAPL",
            config={"api_key": "test_key", "api_secret": "test_secret"},
        )

        with patch("requests.get", side_effect=Exception("API Error")):
            quotes = adapter.fetch_historical_quotes("AAPL", "2025-01-01", "2025-01-02")

        assert quotes == []

    def test_fetch_historical_trades_api_error(self) -> None:
        """Test fetch_historical_trades handles API errors gracefully."""
        from unittest.mock import patch

        adapter = AlpacaL2Adapter(
            symbol="AAPL",
            config={"api_key": "test_key", "api_secret": "test_secret"},
        )

        with patch("requests.get", side_effect=Exception("API Error")):
            trades = adapter.fetch_historical_trades("AAPL", "2025-01-01", "2025-01-02")

        assert trades == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
