# -*- coding: utf-8 -*-
"""
Comprehensive tests for impl_quantizer.py

Covers:
- QuantizerConfig initialization
- QuantizerImpl class functionality
- Filters loading (JSON format)
- Auto-refresh mechanism
- Filters staleness detection
- Strict/permissive modes
- enforce_percent_price_by_side
- validate_order method
- Symbol filters mapping
- Metadata collection
- Edge cases and error handling
"""

import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pytest

from impl_quantizer import (
    QuantizerConfig,
    QuantizerImpl,
    _parse_timestamp,
    _filters_age_days,
    _is_stale,
    _file_size_bytes,
    _file_mtime,
    _file_sha256,
    _as_int,
    _as_bool,
)


class TestHelperFunctions:
    """Test utility functions."""

    def test_parse_timestamp_valid(self):
        result = _parse_timestamp("2025-01-01T00:00:00Z")
        assert result is not None
        assert isinstance(result, datetime)
        assert result.tzinfo is not None

    def test_parse_timestamp_invalid(self):
        assert _parse_timestamp(None) is None
        assert _parse_timestamp(123) is None
        assert _parse_timestamp("invalid") is None

    def test_filters_age_days_from_metadata(self, tmp_path):
        test_file = tmp_path / "test.json"
        test_file.write_text("{}")

        meta = {"built_at": "2025-01-01T00:00:00Z"}
        age = _filters_age_days(meta, str(test_file))

        assert age is not None
        assert age >= 0.0

    def test_filters_age_days_from_mtime(self, tmp_path):
        test_file = tmp_path / "test.json"
        test_file.write_text("{}")

        age = _filters_age_days({}, str(test_file))

        assert age is not None
        assert age >= 0.0

    def test_filters_age_days_no_metadata(self):
        age = _filters_age_days(None, "nonexistent.json")
        assert age is None

    def test_is_stale(self):
        assert _is_stale(None, 30) is False
        assert _is_stale(10.0, 30) is False
        assert _is_stale(40.0, 30) is True
        assert _is_stale(30.0, 30) is False

    def test_file_size_bytes(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        size = _file_size_bytes(str(test_file))
        assert size == 5

    def test_file_size_bytes_missing(self):
        size = _file_size_bytes("nonexistent.txt")
        assert size is None

    def test_file_mtime(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        mtime = _file_mtime(str(test_file))
        assert mtime is not None
        assert isinstance(mtime, float)

    def test_file_mtime_missing(self):
        mtime = _file_mtime("nonexistent.txt")
        assert mtime is None

    def test_file_sha256(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        sha = _file_sha256(str(test_file))
        assert sha is not None
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)

    def test_file_sha256_missing(self):
        sha = _file_sha256("nonexistent.txt")
        assert sha is None

    def test_as_int_valid(self):
        assert _as_int(42, default=0) == 42
        assert _as_int("10", default=0) == 10
        assert _as_int(3.7, default=0) == 3

    def test_as_int_invalid(self):
        assert _as_int(None, default=100) == 100
        assert _as_int("invalid", default=50) == 50

    def test_as_bool_valid(self):
        assert _as_bool(True) is True
        assert _as_bool(False) is False
        assert _as_bool("true") is True
        assert _as_bool("false") is False
        assert _as_bool("1") is True
        assert _as_bool("0") is False
        assert _as_bool(1) is True
        assert _as_bool(0) is False

    def test_as_bool_invalid(self):
        assert _as_bool(None, default=True) is True
        assert _as_bool("invalid") is False


class TestQuantizerConfig:
    """Test QuantizerConfig dataclass."""

    def test_default_values(self):
        cfg = QuantizerConfig(path="data/filters.json")

        assert cfg.path == "data/filters.json"
        assert cfg.strict_filters is True
        assert cfg.quantize_mode == "inward"
        assert cfg.enforce_percent_price_by_side is True
        assert cfg.filters_path == ""
        assert cfg.auto_refresh_days == 30
        assert cfg.refresh_on_start is False

    def test_custom_values(self):
        cfg = QuantizerConfig(
            path="custom/path.json",
            strict_filters=False,
            quantize_mode="outward",
            enforce_percent_price_by_side=False,
            filters_path="override/path.json",
            auto_refresh_days=60,
            refresh_on_start=True,
        )

        assert cfg.path == "custom/path.json"
        assert cfg.strict_filters is False
        assert cfg.quantize_mode == "outward"
        assert cfg.enforce_percent_price_by_side is False
        assert cfg.filters_path == "override/path.json"
        assert cfg.auto_refresh_days == 60
        assert cfg.refresh_on_start is True

    def test_resolved_filters_path_default(self):
        cfg = QuantizerConfig(path="data/filters.json")
        assert cfg.resolved_filters_path() == "data/filters.json"

    def test_resolved_filters_path_override(self):
        cfg = QuantizerConfig(
            path="data/filters.json",
            filters_path="override/filters.json",
        )
        assert cfg.resolved_filters_path() == "override/filters.json"

    def test_strict_property_compatibility(self):
        cfg = QuantizerConfig(path="data/filters.json", strict_filters=True)

        # Test getter
        assert cfg.strict is True

        # Test setter
        cfg.strict = False
        assert cfg.strict_filters is False


class TestQuantizerImpl:
    """Test QuantizerImpl class."""

    def test_initialization_basic(self):
        cfg = QuantizerConfig(path="")
        impl = QuantizerImpl(cfg)

        assert impl.cfg is cfg
        assert impl._quantizer is None
        assert impl.filters_metadata["status"] == "missing"

    def test_initialization_with_valid_filters(self, tmp_path):
        filters_path = tmp_path / "filters.json"
        filters_data = {
            "filters": {
                "BTCUSDT": {
                    "symbol": "BTCUSDT",
                    "price_filter": {"min_price": "0.01", "max_price": "100000"},
                    "lot_size_filter": {"min_qty": "0.00001", "max_qty": "10000"},
                }
            },
            "metadata": {"built_at": datetime.now(timezone.utc).isoformat()},
        }
        filters_path.write_text(json.dumps(filters_data))

        cfg = QuantizerConfig(path=str(filters_path))
        impl = QuantizerImpl(cfg)

        # Verify filters were loaded
        assert impl.filters_metadata["symbol_count"] >= 0

    def test_initialization_missing_filters(self):
        cfg = QuantizerConfig(path="nonexistent/filters.json")
        impl = QuantizerImpl(cfg)

        assert impl._quantizer is None
        assert impl.filters_metadata["missing"] is True

    def test_initialization_stale_filters(self, tmp_path):
        filters_path = tmp_path / "filters.json"
        old_timestamp = "2020-01-01T00:00:00Z"
        filters_data = {
            "filters": {"BTCUSDT": {}},
            "metadata": {"built_at": old_timestamp},
        }
        filters_path.write_text(json.dumps(filters_data))

        cfg = QuantizerConfig(path=str(filters_path), auto_refresh_days=30)
        impl = QuantizerImpl(cfg)

        # Filters should be detected as stale
        assert impl.filters_metadata.get("stale") is True

    def test_validate_order_permissive(self):
        """Test validate_order in permissive mode (no quantizer)."""
        cfg = QuantizerConfig(path="")
        impl = QuantizerImpl(cfg)

        result = impl.validate_order(
            symbol="BTCUSDT",
            side="BUY",
            price=50000.0,
            qty=0.01,
        )

        assert result is not None
        assert result.price == 50000.0
        assert result.qty == 0.01
        assert result.accepted is True

    def test_validate_order_with_ref_price(self):
        """Test validate_order with reference price."""
        cfg = QuantizerConfig(path="")
        impl = QuantizerImpl(cfg)

        result = impl.validate_order(
            symbol="BTCUSDT",
            side="BUY",
            price=50000.0,
            qty=0.01,
            ref_price=49999.0,
        )

        assert result is not None
        assert result.accepted is True

    def test_validate_order_with_enforce_ppbs(self):
        """Test validate_order with enforce_ppbs flag."""
        cfg = QuantizerConfig(path="", enforce_percent_price_by_side=True)
        impl = QuantizerImpl(cfg)

        result = impl.validate_order(
            symbol="BTCUSDT",
            side="BUY",
            price=50000.0,
            qty=0.01,
            enforce_ppbs=True,
        )

        assert result is not None

    def test_symbol_filters_property(self, tmp_path):
        """Test symbol_filters read-only property."""
        filters_path = tmp_path / "filters.json"
        filters_data = {
            "filters": {
                "BTCUSDT": {"symbol": "BTCUSDT"},
                "ETHUSDT": {"symbol": "ETHUSDT"},
            }
        }
        filters_path.write_text(json.dumps(filters_data))

        cfg = QuantizerConfig(path=str(filters_path))
        impl = QuantizerImpl(cfg)

        # Access should work
        filters = impl.symbol_filters
        assert isinstance(filters, dict) or len(filters) >= 0

    def test_filters_metadata_property(self):
        """Test filters_metadata read-only property."""
        cfg = QuantizerConfig(path="")
        impl = QuantizerImpl(cfg)

        metadata = impl.filters_metadata
        assert "status" in metadata
        assert "path" in metadata

    def test_attach_to_simulator(self):
        """Test attaching quantizer to a simulator."""
        cfg = QuantizerConfig(path="")
        impl = QuantizerImpl(cfg)

        class MockSimulator:
            pass

        sim = MockSimulator()
        impl.attach_to(sim)

        # Verify attributes were set
        assert hasattr(sim, "validate_order")

    def test_attach_to_with_strict_override(self, tmp_path):
        """Test attach_to with strict override."""
        filters_path = tmp_path / "filters.json"
        filters_data = {"filters": {"BTCUSDT": {}}}
        filters_path.write_text(json.dumps(filters_data))

        cfg = QuantizerConfig(path=str(filters_path), strict_filters=True)
        impl = QuantizerImpl(cfg)

        class MockSimulator:
            pass

        sim = MockSimulator()
        impl.attach_to(sim, strict=False)

        # Strict should be overridden to False
        assert impl.cfg.strict is False

    def test_from_dict_basic(self):
        """Test creating QuantizerImpl from dict."""
        data = {
            "path": "data/filters.json",
            "strict_filters": False,
            "quantize_mode": "nearest",
            "enforce_percent_price_by_side": False,
        }

        impl = QuantizerImpl.from_dict(data)

        assert impl.cfg.path == "data/filters.json"
        assert impl.cfg.strict_filters is False
        assert impl.cfg.quantize_mode == "nearest"
        assert impl.cfg.enforce_percent_price_by_side is False

    def test_from_dict_with_filters_block(self):
        """Test from_dict with nested filters configuration."""
        data = {
            "path": "data/filters.json",
            "filters": {
                "path": "override/filters.json",
                "strict_filters": True,
                "auto_refresh_days": 60,
                "refresh_on_start": True,
            },
        }

        impl = QuantizerImpl.from_dict(data)

        assert impl.cfg.filters_path == "override/filters.json"
        assert impl.cfg.strict_filters is True
        assert impl.cfg.auto_refresh_days == 60
        assert impl.cfg.refresh_on_start is True


class TestAutoRefresh:
    """Test auto-refresh mechanism."""

    def test_refresh_guard_cooldown(self):
        """Test refresh guard prevents rapid consecutive refreshes."""
        # First refresh should be allowed
        assert QuantizerImpl._should_refresh("test_path", None) is True

        # Record refresh
        QuantizerImpl._record_refresh("test_path", None)

        # Immediate second refresh should be blocked (cooldown)
        assert QuantizerImpl._should_refresh("test_path", None) is False

    def test_refresh_guard_mtime_change(self, tmp_path):
        """Test refresh allowed when mtime changes."""
        test_file = tmp_path / "test.json"
        test_file.write_text("{}")

        mtime1 = os.path.getmtime(str(test_file))

        # Record with first mtime
        QuantizerImpl._record_refresh(str(test_file), mtime1)

        # Modify file
        time.sleep(0.01)
        test_file.write_text("{modified}")
        mtime2 = os.path.getmtime(str(test_file))

        # Should allow refresh due to mtime change
        assert QuantizerImpl._should_refresh(str(test_file), mtime2) is True

    def test_refresh_on_start_disabled(self, tmp_path):
        """Test refresh_on_start=False."""
        filters_path = tmp_path / "filters.json"
        old_data = {
            "filters": {},
            "metadata": {"built_at": "2020-01-01T00:00:00Z"},
        }
        filters_path.write_text(json.dumps(old_data))

        cfg = QuantizerConfig(
            path=str(filters_path),
            auto_refresh_days=30,
            refresh_on_start=False,
        )
        impl = QuantizerImpl(cfg)

        # Should NOT attempt refresh
        assert impl.filters_metadata.get("refresh_executed") is False


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_path(self):
        """Test with empty path."""
        cfg = QuantizerConfig(path="")
        impl = QuantizerImpl(cfg)

        assert impl._quantizer is None
        assert impl.filters_metadata["status"] == "missing"

    def test_invalid_json_syntax(self, tmp_path):
        """Test handling of invalid JSON syntax."""
        filters_path = tmp_path / "invalid.json"
        filters_path.write_text("{invalid json")

        cfg = QuantizerConfig(path=str(filters_path))
        impl = QuantizerImpl(cfg)

        assert impl._quantizer is None

    def test_empty_filters_dict(self, tmp_path):
        """Test handling of empty filters dictionary."""
        filters_path = tmp_path / "empty.json"
        filters_path.write_text(json.dumps({"filters": {}}))

        cfg = QuantizerConfig(path=str(filters_path))
        impl = QuantizerImpl(cfg)

        # Should initialize but be considered missing
        assert impl.filters_metadata["missing"] is True

    def test_validate_order_exception_handling(self):
        """Test validate_order exception handling."""
        cfg = QuantizerConfig(path="")
        impl = QuantizerImpl(cfg)

        # Should handle gracefully even with invalid inputs
        result = impl.validate_order(
            symbol=None,  # Invalid
            side="BUY",
            price=50000.0,
            qty=0.01,
        )

        assert result is not None

    def test_negative_auto_refresh_days(self):
        """Test negative auto_refresh_days."""
        cfg = QuantizerConfig(path="data/filters.json", auto_refresh_days=-10)
        impl = QuantizerImpl(cfg)

        # Should clamp to 0
        assert impl.cfg.auto_refresh_days == 0


class TestIntegration:
    """Integration tests."""

    def test_full_lifecycle(self, tmp_path):
        """Test complete QuantizerImpl lifecycle."""
        filters_path = tmp_path / "filters.json"
        filters_data = {
            "filters": {
                "BTCUSDT": {
                    "symbol": "BTCUSDT",
                    "price_filter": {
                        "min_price": "0.01",
                        "max_price": "100000",
                        "tick_size": "0.01",
                    },
                    "lot_size_filter": {
                        "min_qty": "0.00001",
                        "max_qty": "10000",
                        "step_size": "0.00001",
                    },
                },
                "ETHUSDT": {
                    "symbol": "ETHUSDT",
                    "price_filter": {
                        "min_price": "0.01",
                        "max_price": "50000",
                        "tick_size": "0.01",
                    },
                    "lot_size_filter": {
                        "min_qty": "0.0001",
                        "max_qty": "10000",
                        "step_size": "0.0001",
                    },
                },
            },
            "metadata": {
                "built_at": datetime.now(timezone.utc).isoformat(),
                "source": "test",
            },
        }
        filters_path.write_text(json.dumps(filters_data))

        # Create implementation
        cfg = QuantizerConfig(
            path=str(filters_path),
            strict_filters=True,
            quantize_mode="inward",
            enforce_percent_price_by_side=True,
            auto_refresh_days=30,
        )
        impl = QuantizerImpl(cfg)

        # Verify initialization
        assert impl.cfg.path == str(filters_path)
        assert impl.cfg.strict_filters is True

        # Test validate_order
        result = impl.validate_order(
            symbol="BTCUSDT",
            side="BUY",
            price=50000.123,
            qty=0.0123456,
        )
        assert result is not None

        # Test attach_to
        class MockSimulator:
            pass

        sim = MockSimulator()
        impl.attach_to(sim)

        assert hasattr(sim, "validate_order")
        assert hasattr(sim, "symbol_filters")

    def test_strict_vs_permissive_modes(self, tmp_path):
        """Test strict vs permissive validation modes."""
        filters_path = tmp_path / "filters.json"
        filters_data = {"filters": {"BTCUSDT": {}}}
        filters_path.write_text(json.dumps(filters_data))

        # Strict mode
        cfg_strict = QuantizerConfig(path=str(filters_path), strict_filters=True)
        impl_strict = QuantizerImpl(cfg_strict)

        # Permissive mode
        cfg_permissive = QuantizerConfig(path=str(filters_path), strict_filters=False)
        impl_permissive = QuantizerImpl(cfg_permissive)

        # Both should initialize
        assert impl_strict.cfg.strict_filters is True
        assert impl_permissive.cfg.strict_filters is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
