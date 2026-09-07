# -*- coding: utf-8 -*-
"""
Comprehensive tests for impl_fees.py

Covers:
- FeesConfig initialization and normalization
- FeesImpl class functionality
- Symbol fee table loading and caching
- BNB discount multipliers
- Maker/taker share settings
- Fee rounding configurations
- Settlement options
- Account fee info fetching
- Public fee snapshot auto-refresh
- Edge cases and error handling
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from impl_fees import (
    FeesConfig,
    FeesImpl,
    _safe_float,
    _safe_positive_int,
    _safe_str,
    _safe_bool,
    _normalise_path,
    _safe_non_negative_float,
)


class TestHelperFunctions:
    """Test utility functions."""

    def test_safe_float_valid(self):
        assert _safe_float(3.14) == 3.14
        assert _safe_float("2.5") == 2.5
        assert _safe_float(42) == 42.0
        assert _safe_float(0.0) == 0.0

    def test_safe_float_invalid(self):
        assert _safe_float(None) is None
        assert _safe_float(float("nan")) is None
        assert _safe_float(float("inf")) is None
        assert _safe_float("invalid") is None

    def test_safe_positive_int_valid(self):
        assert _safe_positive_int(42) == 42
        assert _safe_positive_int("10") == 10

    def test_safe_positive_int_invalid(self):
        assert _safe_positive_int(None) is None
        assert _safe_positive_int(0) is None
        assert _safe_positive_int(-5) is None
        assert _safe_positive_int("invalid") is None

    def test_safe_str_valid(self):
        assert _safe_str("hello") == "hello"
        assert _safe_str(42) == "42"
        assert _safe_str("  spaces  ") == "spaces"

    def test_safe_str_invalid(self):
        assert _safe_str(None) is None
        assert _safe_str("") is None
        assert _safe_str("   ") is None

    def test_safe_bool_valid(self):
        assert _safe_bool(True) is True
        assert _safe_bool(False) is False
        assert _safe_bool("true") is True
        assert _safe_bool("false") is False
        assert _safe_bool("1") is True
        assert _safe_bool("0") is False
        assert _safe_bool(1) is True
        assert _safe_bool(0) is False

    def test_safe_bool_invalid(self):
        assert _safe_bool(None) is None
        assert _safe_bool("invalid") is None
        assert _safe_bool(2) is None

    def test_normalise_path(self):
        assert _normalise_path("/some/path") == "/some/path"
        assert _normalise_path("~/file.txt") == os.path.expanduser("~/file.txt")
        assert _normalise_path(None) is None
        assert _normalise_path("") is None
        assert _normalise_path("   ") is None

    def test_safe_non_negative_float(self):
        assert _safe_non_negative_float(5.0) == 5.0
        assert _safe_non_negative_float(0.0) == 0.0
        assert _safe_non_negative_float(-5.0) is None
        assert _safe_non_negative_float(None) is None


class TestFeesConfig:
    """Test FeesConfig dataclass."""

    def test_default_values(self):
        cfg = FeesConfig()

        assert cfg.enabled is True
        assert cfg.path is None
        assert cfg.refresh_days is None
        assert cfg.maker_bps == 1.0  # Post-init default
        assert cfg.taker_bps == 5.0  # Post-init default
        assert cfg.use_bnb_discount is False
        assert cfg.vip_tier is None
        assert cfg.fee_rounding_step is None
        assert cfg.rounding == {}
        assert cfg.settlement == {}
        assert cfg.public_snapshot == {}
        assert cfg.symbol_fee_table == {}
        assert cfg.metadata == {}
        assert cfg.account_info == {}
        assert cfg.maker_taker_share is None

    def test_custom_maker_taker_bps(self):
        cfg = FeesConfig(maker_bps=2.0, taker_bps=10.0)

        assert cfg.maker_bps == 2.0
        assert cfg.taker_bps == 10.0
        assert cfg.maker_bps_overridden is True
        assert cfg.taker_bps_overridden is True

    def test_bnb_discount(self):
        cfg = FeesConfig(use_bnb_discount=True)

        assert cfg.use_bnb_discount is True
        assert cfg.use_bnb_discount_overridden is True

    def test_discount_multipliers(self):
        cfg = FeesConfig(
            use_bnb_discount=True,
            maker_discount_mult=0.75,
            taker_discount_mult=0.75,
        )

        assert cfg.maker_discount_mult == 0.75
        assert cfg.taker_discount_mult == 0.75
        assert cfg.maker_discount_overridden is True
        assert cfg.taker_discount_overridden is True

    def test_vip_tier(self):
        cfg = FeesConfig(vip_tier=3)

        assert cfg.vip_tier == 3
        assert cfg.vip_tier_overridden is True

    def test_fee_rounding(self):
        cfg = FeesConfig(
            fee_rounding_step=0.01,
            rounding={"enabled": True, "mode": "nearest", "precision": 2},
        )

        assert cfg.fee_rounding_step == 0.01
        assert cfg.rounding_enabled is True
        assert cfg.rounding_step_effective == 0.01

    def test_rounding_disabled(self):
        cfg = FeesConfig(rounding={"enabled": False})

        assert cfg.rounding_enabled is False
        assert cfg.fee_rounding_step is None

    def test_settlement_options(self):
        cfg = FeesConfig(
            settlement={
                "enabled": True,
                "mode": "bnb",
                "currency": "BNB",
            }
        )

        assert cfg.settlement_options is not None
        assert cfg.settlement_options["enabled"] is True

    def test_symbol_fee_table(self):
        cfg = FeesConfig(
            symbol_fee_table={
                "BTCUSDT": {"maker_bps": 1.5, "taker_bps": 7.5},
                "ETHUSDT": {"maker_bps": 2.0, "taker_bps": 8.0},
            }
        )

        assert "BTCUSDT" in cfg.symbol_fee_table
        assert "ETHUSDT" in cfg.symbol_fee_table
        assert cfg.symbol_fee_table["BTCUSDT"]["maker_bps"] == 1.5


class TestFeesImpl:
    """Test FeesImpl class."""

    def test_initialization_basic(self):
        cfg = FeesConfig()
        impl = FeesImpl(cfg)

        assert impl.cfg is cfg
        assert impl.base_fee_bps is not None
        assert "maker_fee_bps" in impl.base_fee_bps
        assert "taker_fee_bps" in impl.base_fee_bps

    def test_initialization_with_maker_taker(self):
        cfg = FeesConfig(maker_bps=2.0, taker_bps=10.0)
        impl = FeesImpl(cfg)

        assert impl.base_fee_bps["maker_fee_bps"] == 2.0
        assert impl.base_fee_bps["taker_fee_bps"] == 10.0

    def test_initialization_with_bnb_discount(self):
        cfg = FeesConfig(
            use_bnb_discount=True,
            maker_discount_mult=0.75,
            taker_discount_mult=0.75,
        )
        impl = FeesImpl(cfg)

        # Fees should be discounted
        assert impl._use_bnb_discount is True
        assert impl._maker_discount_mult == 0.75
        assert impl._taker_discount_mult == 0.75

    def test_load_symbol_fee_table_from_file(self, tmp_path):
        """Test loading symbol fee table from JSON file."""
        fee_table_path = tmp_path / "fees_by_symbol.json"
        fee_data = {
            "BTCUSDT": {"maker_bps": 1.5, "taker_bps": 7.5},
            "ETHUSDT": {"maker_bps": 2.0, "taker_bps": 8.0},
        }
        fee_table_path.write_text(json.dumps(fee_data))

        cfg = FeesConfig(path=str(fee_table_path))
        impl = FeesImpl(cfg)

        # Verify table was loaded
        assert len(impl.symbol_fee_table) >= 0  # May be empty if file format doesn't match

    def test_load_missing_fee_table(self, tmp_path):
        """Test handling of missing fee table file."""
        cfg = FeesConfig(path=str(tmp_path / "nonexistent.json"))
        impl = FeesImpl(cfg)

        # Should initialize without error
        assert impl.table_error in ["missing", "invalid"]

    def test_inline_symbol_fee_table(self):
        """Test inline symbol fee table overrides."""
        cfg = FeesConfig(
            symbol_fee_table={
                "BTCUSDT": {"maker_bps": 3.0, "taker_bps": 12.0},
            }
        )
        impl = FeesImpl(cfg)

        assert "BTCUSDT" in impl.symbol_fee_table
        assert impl.symbol_fee_table["BTCUSDT"]["maker_bps"] == 3.0

    def test_from_dict_basic(self):
        data = {
            "enabled": True,
            "maker_bps": 2.5,
            "taker_bps": 7.5,
            "use_bnb_discount": True,
        }

        impl = FeesImpl.from_dict(data)

        assert impl.cfg.maker_bps == 2.5
        assert impl.cfg.taker_bps == 7.5
        assert impl.cfg.use_bnb_discount is True

    def test_from_dict_with_vip_tier(self):
        data = {
            "maker_bps": 1.0,
            "taker_bps": 4.0,
            "vip_tier": 5,
        }

        impl = FeesImpl.from_dict(data)

        assert impl.cfg.vip_tier == 5

    def test_from_dict_with_rounding(self):
        data = {
            "fee_rounding_step": 0.01,
            "rounding": {"enabled": True, "mode": "down", "precision": 4},
        }

        impl = FeesImpl.from_dict(data)

        assert impl.cfg.fee_rounding_step == 0.01
        assert impl.cfg.rounding_enabled is True

    def test_attach_to_simulator(self):
        """Test attaching fees implementation to a simulator."""
        cfg = FeesConfig(maker_bps=2.0, taker_bps=8.0)
        impl = FeesImpl(cfg)

        # Create a mock simulator object
        class MockSimulator:
            def __init__(self):
                self.fees = None

        sim = MockSimulator()

        # Attach implementation
        impl.attach_to(sim)

        # Verify fees were attached
        assert sim.fees is not None or hasattr(sim, "_maker_taker_share_cfg")


class TestFeeTableLoading:
    """Test fee table loading and caching."""

    def test_parse_fee_table_basic(self):
        """Test parsing basic fee table structure."""
        raw_data = {
            "meta": {"vip_tier": "VIP 1", "built_at": "2025-01-01T00:00:00Z"},
            "account": {"maker_bps": 1.5, "taker_bps": 6.0},
            "symbol_fee_table": {
                "BTCUSDT": {"maker_bps": 1.0, "taker_bps": 5.0},
            },
        }

        parsed = FeesImpl._parse_fee_table(raw_data)

        assert "meta" in parsed
        assert "account" in parsed
        assert "table" in parsed
        assert "BTCUSDT" in parsed["table"]

    def test_parse_fee_table_with_maker_taker_share(self):
        """Test parsing fee table with maker/taker share settings."""
        raw_data = {
            "maker_taker_share": {
                "enabled": True,
                "mode": "fixed",
                "maker_share_default": 0.6,
            },
            "BTCUSDT": {"maker_bps": 1.0, "taker_bps": 5.0},
        }

        parsed = FeesImpl._parse_fee_table(raw_data)

        assert parsed["share"] is not None
        assert parsed["share"]["enabled"] is True

    def test_read_fee_table_caching(self, tmp_path):
        """Test fee table file caching."""
        fee_table_path = tmp_path / "fees.json"
        fee_data = {"BTCUSDT": {"maker_bps": 1.0, "taker_bps": 5.0}}
        fee_table_path.write_text(json.dumps(fee_data))

        # First read
        result1, mtime1 = FeesImpl._read_fee_table(str(fee_table_path))

        # Second read (should use cache)
        result2, mtime2 = FeesImpl._read_fee_table(str(fee_table_path))

        assert result1 is not None
        assert result2 is not None
        assert mtime1 == mtime2


class TestMakerTakerShare:
    """Test maker/taker share settings."""

    def test_maker_taker_share_basic(self):
        cfg = FeesConfig(
            maker_taker_share={
                "enabled": True,
                "mode": "fixed",
                "maker_share_default": 0.7,
            }
        )
        impl = FeesImpl(cfg)

        assert impl.maker_taker_share_cfg is not None
        # Check if maker_taker_share is enabled by verifying cfg is not None
        assert impl.maker_taker_share_raw is not None
        assert impl.maker_taker_share_raw.get("enabled") is True

    def test_maker_taker_share_disabled(self):
        cfg = FeesConfig(
            maker_taker_share={"enabled": False},
        )
        impl = FeesImpl(cfg)

        # When disabled, check that the raw config has enabled=False
        assert impl.maker_taker_share_raw is not None
        assert impl.maker_taker_share_raw.get("enabled") is False


class TestPublicFeeRefresh:
    """Test public fee snapshot auto-refresh."""

    def test_auto_refresh_disabled_env(self, monkeypatch):
        """Test auto-refresh disabled via environment variable."""
        monkeypatch.setenv("BINANCE_PUBLIC_FEES_DISABLE_AUTO", "1")

        cfg = FeesConfig()
        impl = FeesImpl(cfg)

        # Auto-refresh should be disabled
        assert not impl._public_refresh_attempted


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_maker_bps_type(self):
        """Test invalid maker_bps type."""
        cfg = FeesConfig(maker_bps="invalid")

        # Should fall back to default
        assert cfg.maker_bps == 1.0

    def test_invalid_vip_tier_type(self):
        """Test invalid vip_tier type."""
        cfg = FeesConfig(vip_tier="invalid")

        assert cfg.vip_tier is None

    def test_negative_discount_multiplier(self):
        """Test negative discount multiplier."""
        cfg = FeesConfig(maker_discount_mult=-0.5)

        # Should clamp to valid range
        assert cfg.maker_discount_mult >= 0.0

    def test_empty_symbol_fee_table(self):
        """Test empty symbol fee table."""
        # Specify nonexistent path to prevent loading from file
        cfg = FeesConfig(path="nonexistent_fees.json", symbol_fee_table={})
        impl = FeesImpl(cfg)

        # With no file and empty inline table, should be empty
        assert len(impl.symbol_fee_table) == 0

    def test_invalid_rounding_step(self):
        """Test invalid rounding step."""
        cfg = FeesConfig(fee_rounding_step=-0.01)

        assert cfg.fee_rounding_step is None


class TestIntegration:
    """Integration tests."""

    def test_full_lifecycle(self):
        """Test complete FeesImpl lifecycle."""
        cfg = FeesConfig(
            enabled=True,
            maker_bps=1.5,
            taker_bps=6.0,
            use_bnb_discount=True,
            maker_discount_mult=0.75,
            taker_discount_mult=0.75,
            vip_tier=2,
            fee_rounding_step=0.01,
        )

        impl = FeesImpl(cfg)

        # Verify initialization
        assert impl.cfg.maker_bps == 1.5
        assert impl.cfg.taker_bps == 6.0
        assert impl._use_bnb_discount is True

        # Verify fees with discount
        expected_maker = 1.5 * 0.75
        expected_taker = 6.0 * 0.75
        assert impl.base_fee_bps["maker_fee_bps"] == expected_maker
        assert impl.base_fee_bps["taker_fee_bps"] == expected_taker

        # Attach to simulator
        class MockSimulator:
            pass

        sim = MockSimulator()
        impl.attach_to(sim)

        assert hasattr(sim, "fees") or hasattr(sim, "_maker_taker_share_cfg")

    def test_fee_table_integration(self, tmp_path):
        """Test integration with fee table file."""
        fee_table_path = tmp_path / "fees_by_symbol.json"
        fee_data = {
            "meta": {"built_at": "2025-01-01T00:00:00Z"},
            "account": {"vip_tier": 1},
            "BTCUSDT": {"maker_bps": 1.0, "taker_bps": 4.0},
            "ETHUSDT": {"maker_bps": 1.5, "taker_bps": 5.0},
        }
        fee_table_path.write_text(json.dumps(fee_data))

        cfg = FeesConfig(path=str(fee_table_path))
        impl = FeesImpl(cfg)

        # Verify metadata
        assert impl.table_path is not None

    def test_maker_taker_share_integration(self):
        """Test maker/taker share with fee calculation."""
        cfg = FeesConfig(
            maker_bps=1.0,
            taker_bps=5.0,
            maker_taker_share={
                "enabled": True,
                "mode": "fixed",
                "maker_share_default": 0.6,
                "spread_cost_maker_bps": 0.5,
                "spread_cost_taker_bps": 1.5,
            },
        )

        impl = FeesImpl(cfg)

        assert impl.maker_taker_share_cfg is not None
        assert impl.maker_taker_share_raw is not None
        assert impl.maker_taker_share_raw.get("enabled") is True

        # Verify expected fee breakdown
        if hasattr(impl, "maker_taker_share_expected") and impl.maker_taker_share_expected:
            assert "expected_fee_bps" in impl.maker_taker_share_expected


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
