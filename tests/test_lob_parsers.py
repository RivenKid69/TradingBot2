"""
Tests for LOBSTER and ITCH parsers.

Tests cover:
- LOBSTER message parsing
- LOBSTER orderbook file parsing
- ITCH binary message parsing
- Error handling for malformed data
"""

import io
import struct
import tempfile
import pytest

pytest.importorskip("sortedcontainers")

from lob.parsers import (
    LOBSTERParser,
    LOBSTERMessage,
    LOBSTEREventType,
    ITCHParser,
    ITCHAddOrder,
    ITCHOrderCancel,
    ITCHOrderDelete,
    ITCHOrderExecuted,
    ITCHOrderReplace,
)
from lob.data_structures import Side


# ==============================================================================
# LOBSTER Parser Tests
# ==============================================================================


class TestLOBSTERParser:
    """Tests for LOBSTER message parser."""

    def test_parse_add_message(self):
        """Test parsing ADD message."""
        parser = LOBSTERParser()
        line = "34200.123456789,1,12345,100,1000000,1"

        msg = parser.parse_line(line)

        assert msg is not None
        assert msg.event_type == LOBSTEREventType.ADD
        assert msg.order_id == 12345
        assert msg.size == 100
        assert msg.price == 1000000  # Raw price
        assert msg.direction == 1
        assert msg.is_buy
        assert msg.is_add

    def test_parse_delete_message(self):
        """Test parsing DELETE message."""
        parser = LOBSTERParser()
        line = "34200.5,3,12345,100,1000000,-1"

        msg = parser.parse_line(line)

        assert msg is not None
        assert msg.event_type == LOBSTEREventType.DELETE
        assert msg.is_delete
        assert not msg.is_buy

    def test_parse_execute_message(self):
        """Test parsing EXECUTE message."""
        parser = LOBSTERParser()
        line = "34200.5,4,12345,50,1000000,1"

        msg = parser.parse_line(line)

        assert msg is not None
        assert msg.event_type == LOBSTEREventType.EXECUTE
        assert msg.is_execute
        assert msg.size == 50

    def test_parse_hidden_execute(self):
        """Test parsing HIDDEN EXECUTE message."""
        parser = LOBSTERParser()
        line = "34200.5,5,12345,50,1000000,1"

        msg = parser.parse_line(line)

        assert msg is not None
        assert msg.event_type == LOBSTEREventType.HIDDEN_EXECUTE
        assert msg.is_execute

    def test_parse_modify_message(self):
        """Test parsing MODIFY message."""
        parser = LOBSTERParser()
        line = "34200.5,2,12345,75,1000000,1"

        msg = parser.parse_line(line)

        assert msg is not None
        assert msg.event_type == LOBSTEREventType.MODIFY
        assert msg.is_modify
        assert msg.size == 75

    def test_parse_cross_message(self):
        """Test parsing CROSS message."""
        parser = LOBSTERParser()
        line = "34200.5,6,12345,100,1000000,1"

        msg = parser.parse_line(line)

        assert msg is not None
        assert msg.event_type == LOBSTEREventType.CROSS

    def test_parse_trade_halt(self):
        """Test parsing TRADE HALT message."""
        parser = LOBSTERParser()
        line = "34200.5,7,0,0,0,0"

        msg = parser.parse_line(line)

        assert msg is not None
        assert msg.event_type == LOBSTEREventType.TRADE_HALT

    def test_price_multiplier(self):
        """Test price multiplier."""
        parser = LOBSTERParser(price_multiplier=0.0001)
        line = "34200.5,1,12345,100,1000000,1"

        msg = parser.parse_line(line)

        assert msg is not None
        assert msg.price == 100.0  # 1000000 * 0.0001

    def test_timestamp_to_ns(self):
        """Test timestamp conversion to nanoseconds."""
        parser = LOBSTERParser()
        line = "34200.123456789,1,12345,100,1000000,1"

        msg = parser.parse_line(line)

        assert msg is not None
        assert msg.timestamp_ns == 34200123456789

    def test_side_property(self):
        """Test side property."""
        parser = LOBSTERParser()

        msg_buy = parser.parse_line("34200.5,1,12345,100,1000000,1")
        assert msg_buy.side == Side.BUY

        msg_sell = parser.parse_line("34200.5,1,12345,100,1000000,-1")
        assert msg_sell.side == Side.SELL

    def test_to_limit_order(self):
        """Test conversion to LimitOrder."""
        parser = LOBSTERParser(price_multiplier=0.0001)
        line = "34200.5,1,12345,100,1000000,1"

        msg = parser.parse_line(line)
        order = msg.to_limit_order(order_id_prefix="AAPL_")

        assert order.order_id == "AAPL_12345"
        assert order.price == 100.0
        assert order.qty == 100.0
        assert order.side == Side.BUY

    def test_parse_malformed_line(self):
        """Test handling malformed lines."""
        parser = LOBSTERParser()

        # Too few fields
        msg = parser.parse_line("34200.5,1,12345")
        assert msg is None

        # Invalid event type
        msg = parser.parse_line("34200.5,99,12345,100,1000000,1")
        assert msg is None

    def test_parse_file(self):
        """Test parsing file."""
        # Create temp file
        content = """34200.1,1,1001,100,1000000,1
34200.2,1,1002,200,1000100,-1
34200.3,4,1001,50,1000000,1
34200.4,3,1001,50,1000000,1"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        parser = LOBSTERParser()
        messages = list(parser.parse_file(filepath))

        assert len(messages) == 4
        assert messages[0].is_add
        assert messages[2].is_execute

    def test_parse_file_max_messages(self):
        """Test max_messages limit."""
        content = "\n".join([f"34200.{i},1,{1000+i},100,1000000,1" for i in range(100)])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        parser = LOBSTERParser()
        messages = list(parser.parse_file(filepath, max_messages=10))

        assert len(messages) == 10

    def test_error_rate(self):
        """Test error rate calculation."""
        parser = LOBSTERParser()

        # Parse mix of valid and invalid
        parser.parse_line("34200.5,1,12345,100,1000000,1")  # Valid
        parser.parse_line("invalid")  # Invalid
        parser.parse_line("34200.5,1,12345,100,1000000,-1")  # Valid
        parser.parse_line("bad,data")  # Invalid

        assert parser.line_count == 4
        assert parser.error_count == 2
        assert parser.error_rate == 0.5

    def test_parse_orderbook_file(self):
        """Test parsing orderbook snapshot file."""
        # Format: ask_price_1, ask_size_1, bid_price_1, bid_size_1, ...
        content = "1000100,100,1000000,200,1000200,150,999900,250"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
            f.flush()
            filepath = f.name

        parser = LOBSTERParser()
        asks, bids = parser.parse_orderbook_file(filepath, n_levels=2)

        assert len(asks) == 2
        assert len(bids) == 2
        assert asks[0] == (1000100, 100)
        assert bids[0] == (1000000, 200)


# ==============================================================================
# ITCH Parser Tests
# ==============================================================================


class TestITCHParser:
    """Tests for ITCH binary parser."""

    def _make_timestamp(self, ns: int) -> bytes:
        """Create 6-byte ITCH timestamp."""
        ts_bytes = struct.pack(">Q", ns)
        return ts_bytes[2:]  # Last 6 bytes

    def _make_add_order(
        self,
        order_ref: int,
        side: str,
        shares: int,
        stock: str,
        price: int,
    ) -> bytes:
        """Create Add Order message (type A)."""
        data = b"A"  # Message type
        data += struct.pack(">H", 1)  # Stock locate
        data += struct.pack(">H", 0)  # Tracking number
        data += self._make_timestamp(1000000000)
        data += struct.pack(">Q", order_ref)
        data += side.encode("ascii")
        data += struct.pack(">I", shares)
        data += stock.ljust(8).encode("ascii")
        data += struct.pack(">I", price)
        return data

    def _make_order_executed(
        self,
        order_ref: int,
        shares: int,
        match_number: int,
    ) -> bytes:
        """Create Order Executed message (type E)."""
        data = b"E"  # Message type
        data += struct.pack(">H", 1)  # Stock locate
        data += struct.pack(">H", 0)  # Tracking number
        data += self._make_timestamp(1000000000)
        data += struct.pack(">Q", order_ref)
        data += struct.pack(">I", shares)
        data += struct.pack(">Q", match_number)
        return data

    def _make_order_cancel(self, order_ref: int, cancelled: int) -> bytes:
        """Create Order Cancel message (type X)."""
        data = b"X"
        data += struct.pack(">H", 1)
        data += struct.pack(">H", 0)
        data += self._make_timestamp(1000000000)
        data += struct.pack(">Q", order_ref)
        data += struct.pack(">I", cancelled)
        return data

    def _make_order_delete(self, order_ref: int) -> bytes:
        """Create Order Delete message (type D)."""
        data = b"D"
        data += struct.pack(">H", 1)
        data += struct.pack(">H", 0)
        data += self._make_timestamp(1000000000)
        data += struct.pack(">Q", order_ref)
        return data

    def _make_order_replace(
        self,
        original_ref: int,
        new_ref: int,
        shares: int,
        price: int,
    ) -> bytes:
        """Create Order Replace message (type U)."""
        data = b"U"
        data += struct.pack(">H", 1)
        data += struct.pack(">H", 0)
        data += self._make_timestamp(1000000000)
        data += struct.pack(">Q", original_ref)
        data += struct.pack(">Q", new_ref)
        data += struct.pack(">I", shares)
        data += struct.pack(">I", price)
        return data

    def test_parse_add_order(self):
        """Test parsing Add Order message."""
        parser = ITCHParser()
        data = self._make_add_order(
            order_ref=12345,
            side="B",
            shares=100,
            stock="AAPL",
            price=1500000,  # $150.00 * 10000
        )

        msg = parser.parse_message(data)

        assert isinstance(msg, ITCHAddOrder)
        assert msg.order_ref == 12345
        assert msg.side == "B"
        assert msg.shares == 100
        assert msg.stock == "AAPL"
        assert msg.price == 150.0  # Converted

    def test_parse_order_executed(self):
        """Test parsing Order Executed message."""
        parser = ITCHParser()
        data = self._make_order_executed(
            order_ref=12345,
            shares=50,
            match_number=98765,
        )

        msg = parser.parse_message(data)

        assert isinstance(msg, ITCHOrderExecuted)
        assert msg.order_ref == 12345
        assert msg.shares == 50
        assert msg.match_number == 98765

    def test_parse_order_cancel(self):
        """Test parsing Order Cancel message."""
        parser = ITCHParser()
        data = self._make_order_cancel(
            order_ref=12345,
            cancelled=30,
        )

        msg = parser.parse_message(data)

        assert isinstance(msg, ITCHOrderCancel)
        assert msg.order_ref == 12345
        assert msg.cancelled_shares == 30

    def test_parse_order_delete(self):
        """Test parsing Order Delete message."""
        parser = ITCHParser()
        data = self._make_order_delete(order_ref=12345)

        msg = parser.parse_message(data)

        assert isinstance(msg, ITCHOrderDelete)
        assert msg.order_ref == 12345

    def test_parse_order_replace(self):
        """Test parsing Order Replace message."""
        parser = ITCHParser()
        data = self._make_order_replace(
            original_ref=12345,
            new_ref=12346,
            shares=200,
            price=1510000,
        )

        msg = parser.parse_message(data)

        assert isinstance(msg, ITCHOrderReplace)
        assert msg.original_ref == 12345
        assert msg.new_ref == 12346
        assert msg.shares == 200
        assert msg.price == 151.0

    def test_parse_unsupported_message(self):
        """Test unsupported message type returns None."""
        parser = ITCHParser()
        data = b"Z" + b"\x00" * 20  # Unknown type

        msg = parser.parse_message(data)
        assert msg is None

    def test_parse_stream(self):
        """Test parsing message stream."""
        messages = [
            self._make_add_order(1001, "B", 100, "AAPL", 1500000),
            self._make_add_order(1002, "S", 200, "AAPL", 1510000),
            self._make_order_executed(1001, 50, 1),
        ]

        # Build stream with length prefixes
        stream_data = b""
        for msg in messages:
            stream_data += struct.pack(">H", len(msg))
            stream_data += msg

        stream = io.BytesIO(stream_data)
        parser = ITCHParser()

        parsed = list(parser.parse_stream(stream))

        assert len(parsed) == 3
        assert isinstance(parsed[0], ITCHAddOrder)
        assert isinstance(parsed[1], ITCHAddOrder)
        assert isinstance(parsed[2], ITCHOrderExecuted)

    def test_parse_stream_max_messages(self):
        """Test max_messages limit in stream parsing."""
        messages = [self._make_add_order(1000 + i, "B", 100, "AAPL", 1500000) for i in range(10)]

        stream_data = b""
        for msg in messages:
            stream_data += struct.pack(">H", len(msg))
            stream_data += msg

        stream = io.BytesIO(stream_data)
        parser = ITCHParser()

        parsed = list(parser.parse_stream(stream, max_messages=5))

        assert len(parsed) == 5


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestParserIntegration:
    """Integration tests for parsers with OrderBook."""

    def test_lobster_to_orderbook(self):
        """Test building orderbook from LOBSTER messages."""
        from lob.data_structures import OrderBook
        from lob.state_manager import LOBStateManager

        manager = LOBStateManager(symbol="AAPL")
        parser = LOBSTERParser(price_multiplier=0.0001)

        # Simulate message sequence
        messages = [
            "34200.1,1,1001,100,1000000,1",  # Add bid
            "34200.2,1,1002,100,1000100,-1",  # Add ask
            "34200.3,1,1003,50,999900,1",  # Add lower bid
            "34200.4,4,1001,50,1000000,1",  # Execute 50 of bid
        ]

        for line in messages:
            msg = parser.parse_line(line)
            if msg:
                manager.apply_lobster_message(msg)

        book = manager.orderbook
        assert book.best_bid == 100.0
        assert book.best_ask == 100.01

        # Original bid should have 50 remaining
        order = book.get_order("1001")
        assert order is not None
        assert order.remaining_qty == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
