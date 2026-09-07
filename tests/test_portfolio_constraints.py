# -*- coding: utf-8 -*-
"""
Tests for Portfolio-Level Risk Constraints (Task 5).

Tests cover:
1. PositionLimit - Per-symbol weight limits
2. SectorExposure - Sector-level exposure limits
3. FactorTiltLimit - Factor exposure limits
4. PortfolioConstraintManager - Central constraint manager
5. Order validation
6. Portfolio rebalancing
7. Backward compatibility with existing systems

Total: 40+ tests
"""

import pytest
from typing import Dict

from services.portfolio_constraints import (
    ConstraintViolationType,
    FactorTiltLimit,
    FactorTiltValidator,
    Order,
    PortfolioConstraintManager,
    PortfolioState,
    PositionLimit,
    PositionLimitsValidator,
    RebalanceAction,
    RebalanceEngine,
    RebalanceResult,
    SectorExposure,
    SectorExposureValidator,
    ValidationResult,
    create_constraint_manager,
    rebalance_weights,
    validate_order,
    SYMBOL_TO_SECTOR,
)


# ==============================================================================
# Test PositionLimit
# ==============================================================================


class TestPositionLimit:
    """Tests for PositionLimit dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        limit = PositionLimit(symbol="AAPL")
        assert limit.symbol == "AAPL"
        assert limit.max_weight == 0.10
        assert limit.min_weight == 0.0
        assert limit.reason == ""

    def test_custom_values(self) -> None:
        """Test custom values."""
        limit = PositionLimit(
            symbol="MSFT",
            max_weight=0.15,
            min_weight=0.02,
            reason="Core holding",
        )
        assert limit.symbol == "MSFT"
        assert limit.max_weight == 0.15
        assert limit.min_weight == 0.02
        assert limit.reason == "Core holding"

    def test_invalid_max_weight(self) -> None:
        """Test validation of max_weight."""
        with pytest.raises(ValueError, match="max_weight must be in"):
            PositionLimit(symbol="AAPL", max_weight=1.5)

        with pytest.raises(ValueError, match="max_weight must be in"):
            PositionLimit(symbol="AAPL", max_weight=-0.1)

    def test_invalid_min_weight(self) -> None:
        """Test validation of min_weight."""
        # min_weight below -1.0 is invalid (shorts limited to -100%)
        with pytest.raises(ValueError, match="min_weight must be in"):
            PositionLimit(symbol="AAPL", min_weight=-1.5)

        # min_weight > max_weight is invalid
        with pytest.raises(ValueError, match="min_weight must be in"):
            PositionLimit(symbol="AAPL", max_weight=0.10, min_weight=0.15)

    def test_negative_min_weight_allowed(self) -> None:
        """Test that negative min_weight is allowed for shorts."""
        # Should not raise
        limit = PositionLimit(symbol="AAPL", max_weight=0.50, min_weight=-0.20)
        assert limit.min_weight == -0.20

    def test_validate_weight_within_limits(self) -> None:
        """Test weight validation within limits."""
        limit = PositionLimit(symbol="AAPL", max_weight=0.10, min_weight=0.02)

        assert limit.validate_weight(0.05) is None
        assert limit.validate_weight(0.02) is None
        assert limit.validate_weight(0.10) is None

    def test_validate_weight_exceeds_max(self) -> None:
        """Test weight validation exceeding max."""
        limit = PositionLimit(symbol="AAPL", max_weight=0.10)

        violation = limit.validate_weight(0.15)
        assert violation == ConstraintViolationType.POSITION_LIMIT_MAX

    def test_validate_weight_below_min(self) -> None:
        """Test weight validation below min."""
        limit = PositionLimit(symbol="AAPL", min_weight=0.02)

        violation = limit.validate_weight(0.01)
        assert violation == ConstraintViolationType.POSITION_LIMIT_MIN


# ==============================================================================
# Test SectorExposure
# ==============================================================================


class TestSectorExposure:
    """Tests for SectorExposure dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        exposure = SectorExposure(sector="TECHNOLOGY")
        assert exposure.sector == "TECHNOLOGY"
        assert exposure.max_weight == 0.30
        assert exposure.min_weight == 0.0

    def test_custom_values(self) -> None:
        """Test custom values."""
        exposure = SectorExposure(
            sector="FINANCIALS",
            max_weight=0.25,
            min_weight=0.05,
            reason="Underweight financials",
        )
        assert exposure.sector == "FINANCIALS"
        assert exposure.max_weight == 0.25
        assert exposure.min_weight == 0.05

    def test_validation(self) -> None:
        """Test weight validation."""
        exposure = SectorExposure(sector="TECHNOLOGY", max_weight=0.30)

        assert exposure.validate_weight(0.25) is None
        assert exposure.validate_weight(0.35) == ConstraintViolationType.SECTOR_LIMIT_MAX


# ==============================================================================
# Test FactorTiltLimit
# ==============================================================================


class TestFactorTiltLimit:
    """Tests for FactorTiltLimit dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        limit = FactorTiltLimit(factor="beta")
        assert limit.factor == "beta"
        assert limit.max_tilt == 2.0
        assert limit.min_tilt == -2.0

    def test_custom_values(self) -> None:
        """Test custom values."""
        limit = FactorTiltLimit(
            factor="momentum",
            max_tilt=1.5,
            min_tilt=-0.5,
            reason="Limited momentum exposure",
        )
        assert limit.factor == "momentum"
        assert limit.max_tilt == 1.5
        assert limit.min_tilt == -0.5

    def test_validation(self) -> None:
        """Test tilt validation."""
        limit = FactorTiltLimit(factor="beta", max_tilt=1.5, min_tilt=-0.5)

        assert limit.validate_tilt(1.0) is None
        assert limit.validate_tilt(2.0) == ConstraintViolationType.FACTOR_TILT_MAX
        assert limit.validate_tilt(-1.0) == ConstraintViolationType.FACTOR_TILT_MIN


# ==============================================================================
# Test PortfolioState
# ==============================================================================


class TestPortfolioState:
    """Tests for PortfolioState dataclass."""

    def test_compute_weights(self) -> None:
        """Test weight computation from positions."""
        state = PortfolioState(
            positions={
                "AAPL": (100, 150.0),  # $15,000
                "MSFT": (50, 300.0),  # $15,000
            },
            cash=20000.0,
        )

        state.compute_weights()

        # Total NAV = 15000 + 15000 + 20000 = 50000
        assert state.total_nav == 50000.0
        assert state.weights["AAPL"] == pytest.approx(0.30, rel=0.01)
        assert state.weights["MSFT"] == pytest.approx(0.30, rel=0.01)

    def test_compute_sector_weights(self) -> None:
        """Test sector weight computation."""
        state = PortfolioState(
            positions={
                "AAPL": (100, 150.0),  # Technology
                "MSFT": (50, 300.0),  # Technology
                "JPM": (100, 100.0),  # Financials
            },
            cash=0.0,
        )

        state.compute_weights()
        state.compute_sector_weights()

        assert "TECHNOLOGY" in state.sector_weights
        assert "FINANCIALS" in state.sector_weights
        # AAPL (15000) + MSFT (15000) = 30000 of 40000 total = 75% Technology
        assert state.sector_weights["TECHNOLOGY"] == pytest.approx(0.75, rel=0.01)

    def test_empty_portfolio(self) -> None:
        """Test empty portfolio state."""
        state = PortfolioState()
        state.compute_weights()
        assert state.weights == {}


# ==============================================================================
# Test PositionLimitsValidator
# ==============================================================================


class TestPositionLimitsValidator:
    """Tests for PositionLimitsValidator."""

    def test_add_and_get_limit(self) -> None:
        """Test adding and retrieving limits."""
        validator = PositionLimitsValidator()
        limit = PositionLimit(symbol="AAPL", max_weight=0.15)
        validator.add_limit(limit)

        retrieved = validator.get_limit("AAPL")
        assert retrieved.max_weight == 0.15

    def test_default_limit(self) -> None:
        """Test default limit for unknown symbols."""
        validator = PositionLimitsValidator()
        limit = validator.get_limit("UNKNOWN")
        assert limit.max_weight == 1.0  # Default allows full position

    def test_set_default_limit(self) -> None:
        """Test setting custom default limit."""
        validator = PositionLimitsValidator()
        validator.set_default_limit(PositionLimit("__DEFAULT__", max_weight=0.05))

        limit = validator.get_limit("UNKNOWN")
        assert limit.max_weight == 0.05

    def test_validate_weights(self) -> None:
        """Test validating multiple weights."""
        validator = PositionLimitsValidator()
        validator.add_limit(PositionLimit("AAPL", max_weight=0.10))
        validator.add_limit(PositionLimit("MSFT", max_weight=0.15))

        weights = {"AAPL": 0.12, "MSFT": 0.10}  # AAPL exceeds limit
        violations = validator.validate(weights)

        assert len(violations) == 1
        assert violations[0][0] == "AAPL"
        assert violations[0][1] == ConstraintViolationType.POSITION_LIMIT_MAX

    def test_case_insensitive_symbols(self) -> None:
        """Test case insensitivity of symbol lookup."""
        validator = PositionLimitsValidator()
        validator.add_limit(PositionLimit("AAPL", max_weight=0.10))

        limit = validator.get_limit("aapl")
        assert limit.max_weight == 0.10


# ==============================================================================
# Test SectorExposureValidator
# ==============================================================================


class TestSectorExposureValidator:
    """Tests for SectorExposureValidator."""

    def test_add_and_get_limit(self) -> None:
        """Test adding and retrieving sector limits."""
        validator = SectorExposureValidator()
        validator.add_limit(SectorExposure("TECHNOLOGY", max_weight=0.25))

        limit = validator.get_limit("TECHNOLOGY")
        assert limit is not None
        assert limit.max_weight == 0.25

    def test_compute_sector_weights(self) -> None:
        """Test sector weight computation."""
        validator = SectorExposureValidator()
        weights = {"AAPL": 0.30, "MSFT": 0.20, "JPM": 0.50}

        sector_weights = validator.compute_sector_weights(weights)

        assert sector_weights["TECHNOLOGY"] == pytest.approx(0.50, rel=0.01)
        assert sector_weights["FINANCIALS"] == pytest.approx(0.50, rel=0.01)

    def test_validate_sector_weights(self) -> None:
        """Test sector weight validation."""
        validator = SectorExposureValidator()
        validator.add_limit(SectorExposure("TECHNOLOGY", max_weight=0.40))

        weights = {"AAPL": 0.30, "MSFT": 0.20}  # 50% tech, exceeds 40% limit
        violations = validator.validate(weights)

        assert len(violations) == 1
        assert violations[0][0] == "TECHNOLOGY"


# ==============================================================================
# Test FactorTiltValidator
# ==============================================================================


class TestFactorTiltValidator:
    """Tests for FactorTiltValidator."""

    def test_set_and_get_factor_loadings(self) -> None:
        """Test setting and getting factor loadings."""
        validator = FactorTiltValidator()
        validator.set_factor_loadings("AAPL", {"beta": 1.2, "momentum": 0.5})

        assert validator.get_factor_loading("AAPL", "beta") == 1.2
        assert validator.get_factor_loading("AAPL", "momentum") == 0.5
        assert validator.get_factor_loading("AAPL", "unknown") == 0.0

    def test_compute_portfolio_factor_exposure(self) -> None:
        """Test portfolio factor exposure computation."""
        validator = FactorTiltValidator()
        validator.set_factor_loadings("AAPL", {"beta": 1.2})
        validator.set_factor_loadings("MSFT", {"beta": 0.8})

        weights = {"AAPL": 0.50, "MSFT": 0.50}
        exposure = validator.compute_portfolio_factor_exposure(weights, "beta")

        # 0.5 * 1.2 + 0.5 * 0.8 = 1.0
        assert exposure == pytest.approx(1.0, rel=0.01)

    def test_validate_factor_exposure(self) -> None:
        """Test factor exposure validation."""
        validator = FactorTiltValidator()
        validator.add_limit(FactorTiltLimit("beta", max_tilt=1.0, min_tilt=0.5))
        validator.set_factor_loadings("AAPL", {"beta": 1.5})

        weights = {"AAPL": 1.0}  # 100% in high-beta stock
        violations = validator.validate(weights)

        assert len(violations) == 1
        assert violations[0][0] == "beta"


# ==============================================================================
# Test RebalanceEngine
# ==============================================================================


class TestRebalanceEngine:
    """Tests for RebalanceEngine."""

    def test_rebalance_to_equal_weight(self) -> None:
        """Test equal weight rebalancing."""
        engine = RebalanceEngine()
        current = {"AAPL": 0.50, "MSFT": 0.30, "GOOGL": 0.20}
        symbols = ["AAPL", "MSFT", "GOOGL"]

        result = engine.rebalance_to_equal_weight(symbols, current)

        for symbol in symbols:
            assert result.target_weights[symbol] == pytest.approx(1 / 3, rel=0.01)

    def test_rebalance_actions(self) -> None:
        """Test rebalance action computation."""
        engine = RebalanceEngine()
        current = {"AAPL": 0.50, "MSFT": 0.30}
        target = {"AAPL": 0.40, "MSFT": 0.40}

        result = engine.rebalance_to_target(target, current)

        assert result.actions["AAPL"][0] == RebalanceAction.REDUCE
        assert result.actions["MSFT"][0] == RebalanceAction.INCREASE

    def test_rebalance_empty_portfolio(self) -> None:
        """Test rebalancing empty portfolio."""
        engine = RebalanceEngine()
        result = engine.rebalance_to_equal_weight([], {})

        assert result.target_weights == {}
        assert result.estimated_turnover == 0.0


# ==============================================================================
# Test PortfolioConstraintManager
# ==============================================================================


class TestPortfolioConstraintManager:
    """Tests for PortfolioConstraintManager."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        manager = PortfolioConstraintManager()
        assert manager is not None

    def test_add_position_limit(self) -> None:
        """Test adding position limits."""
        manager = PortfolioConstraintManager()
        manager.add_position_limit(PositionLimit("AAPL", max_weight=0.10))

        # Verify by validating
        state = PortfolioState(
            positions={"AAPL": (100, 150.0)},
            cash=1000.0,
            total_nav=16000.0,
        )
        state.compute_weights()
        result = manager.validate_portfolio_state(state)
        # AAPL weight = 15000 / 16000 = 93.75% > 10% limit
        assert not result.is_valid

    def test_add_sector_exposure(self) -> None:
        """Test adding sector exposure limits."""
        manager = PortfolioConstraintManager()
        manager.add_sector_exposure(SectorExposure("TECHNOLOGY", max_weight=0.20))

        state = PortfolioState(
            positions={"AAPL": (100, 150.0), "MSFT": (50, 300.0)},
            total_nav=30000.0,
        )
        state.compute_weights()

        result = manager.validate_portfolio_state(state)
        # Both are tech, 100% > 20%
        assert not result.is_valid

    def test_validate_order_buy(self) -> None:
        """Test order validation for buy orders."""
        manager = PortfolioConstraintManager()
        manager.add_position_limit(PositionLimit("AAPL", max_weight=0.10))

        state = PortfolioState(
            positions={"AAPL": (50, 150.0)},  # $7,500
            cash=92500.0,
            total_nav=100000.0,
        )
        state.compute_weights()

        # Buy 100 shares @ $150 = $15,000 → total = $22,500 = 22.5%
        order = Order(symbol="AAPL", side="BUY", qty=100, price=150.0)
        result = manager.validate_order(order, state)

        assert not result.is_valid
        assert ConstraintViolationType.POSITION_LIMIT_MAX in result.violations

    def test_validate_order_sell(self) -> None:
        """Test order validation for sell orders."""
        manager = PortfolioConstraintManager()
        manager.add_position_limit(PositionLimit("AAPL", max_weight=0.50))

        state = PortfolioState(
            positions={"AAPL": (100, 150.0)},  # $15,000
            cash=15000.0,
            total_nav=30000.0,
        )
        state.compute_weights()

        # Sell 100 shares @ $150 = reduce to 0%
        order = Order(symbol="AAPL", side="SELL", qty=100, price=150.0)
        result = manager.validate_order(order, state)

        assert result.is_valid

    def test_validate_order_adjusted_qty(self) -> None:
        """Test that adjusted quantity is suggested on violation."""
        manager = PortfolioConstraintManager()
        manager.add_position_limit(PositionLimit("AAPL", max_weight=0.10))

        state = PortfolioState(
            positions={},
            cash=100000.0,
            total_nav=100000.0,
        )
        state.compute_weights()

        # Try to buy $20,000 worth (20%) when max is 10%
        order = Order(symbol="AAPL", side="BUY", qty=133.33, price=150.0)  # ~$20,000
        result = manager.validate_order(order, state)

        assert not result.is_valid
        assert result.adjusted_qty is not None
        # Max allowed is 10% = $10,000 = ~66.67 shares
        assert result.adjusted_qty == pytest.approx(66.67, rel=0.1)

    def test_rebalance_weights(self) -> None:
        """Test weight rebalancing."""
        manager = PortfolioConstraintManager()
        manager.add_position_limit(PositionLimit("AAPL", max_weight=0.10))

        current = {"AAPL": 0.50, "MSFT": 0.50}
        adjusted = manager.rebalance_weights(current)

        # AAPL should be capped at 10%
        assert adjusted["AAPL"] == pytest.approx(0.10, rel=0.01)

    def test_get_sector_weights(self) -> None:
        """Test getting sector weights."""
        manager = PortfolioConstraintManager()
        weights = {"AAPL": 0.40, "JPM": 0.30, "XOM": 0.30}

        sector_weights = manager.get_sector_weights(weights)

        assert sector_weights["TECHNOLOGY"] == pytest.approx(0.40, rel=0.01)
        assert sector_weights["FINANCIALS"] == pytest.approx(0.30, rel=0.01)
        assert sector_weights["ENERGY"] == pytest.approx(0.30, rel=0.01)


# ==============================================================================
# Test Factory Functions
# ==============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_constraint_manager(self) -> None:
        """Test create_constraint_manager factory."""
        manager = create_constraint_manager(
            max_position_weight=0.05,
            max_sector_weight=0.20,
        )

        assert manager is not None

    def test_validate_order_convenience(self) -> None:
        """Test validate_order convenience function."""
        state = PortfolioState(
            positions={},
            cash=100000.0,
            total_nav=100000.0,
        )
        state.compute_weights()

        order = Order(symbol="AAPL", side="BUY", qty=10, price=150.0)
        result = validate_order(order, state)

        assert result.is_valid

    def test_rebalance_weights_convenience(self) -> None:
        """Test rebalance_weights convenience function."""
        current = {"AAPL": 0.50, "MSFT": 0.50}
        target = {"AAPL": 0.40, "MSFT": 0.40, "GOOGL": 0.20}

        adjusted = rebalance_weights(current, target)

        assert "GOOGL" in adjusted


# ==============================================================================
# Test Edge Cases
# ==============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_nav_validation(self) -> None:
        """Test validation with zero NAV."""
        manager = PortfolioConstraintManager()

        state = PortfolioState(total_nav=0.0)
        order = Order(symbol="AAPL", side="BUY", qty=10, price=150.0)

        result = manager.validate_order(order, state)
        assert not result.is_valid
        assert "NAV" in result.rejection_reason or "NAV" in str(result.messages)

    def test_no_price_validation(self) -> None:
        """Test validation without price."""
        manager = PortfolioConstraintManager()

        state = PortfolioState(cash=100000.0, total_nav=100000.0)
        state.compute_weights()

        order = Order(symbol="AAPL", side="BUY", qty=10)  # No price
        result = manager.validate_order(order, state)

        assert not result.is_valid

    def test_short_position_validation(self) -> None:
        """Test validation of short positions."""
        manager = PortfolioConstraintManager()
        # Allow shorts by setting negative min_weight
        manager.add_position_limit(PositionLimit("AAPL", max_weight=0.50, min_weight=-0.20))

        state = PortfolioState(
            positions={"AAPL": (10, 150.0)},
            cash=100000.0,
            total_nav=101500.0,
        )
        state.compute_weights()

        # Sell more than we own to go short
        order = Order(symbol="AAPL", side="SELL", qty=30, price=150.0)  # Short 20 shares
        result = manager.validate_order(order, state)

        assert result.is_valid

    def test_unknown_sector_handling(self) -> None:
        """Test handling of symbols with unknown sector."""
        manager = PortfolioConstraintManager()
        weights = {"UNKNOWN_SYMBOL": 1.0}

        sector_weights = manager.get_sector_weights(weights)
        assert "UNKNOWN" in sector_weights
        assert sector_weights["UNKNOWN"] == 1.0


# ==============================================================================
# Test Backward Compatibility
# ==============================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing systems."""

    def test_crypto_symbols_not_in_sector_mapping(self) -> None:
        """Test that crypto symbols (if used) are handled gracefully."""
        manager = PortfolioConstraintManager()

        weights = {"BTCUSDT": 0.50, "ETHUSDT": 0.50}
        sector_weights = manager.get_sector_weights(weights)

        # Both should map to UNKNOWN
        assert sector_weights.get("UNKNOWN", 0.0) == 1.0

    def test_integration_with_order_types(self) -> None:
        """Test that different order types work."""
        manager = PortfolioConstraintManager()

        state = PortfolioState(cash=100000.0, total_nav=100000.0)
        state.compute_weights()

        # MARKET order
        order1 = Order(symbol="AAPL", side="BUY", qty=10, price=150.0, order_type="MARKET")
        result1 = manager.validate_order(order1, state)
        assert result1.is_valid

        # LIMIT order
        order2 = Order(symbol="AAPL", side="BUY", qty=10, price=150.0, order_type="LIMIT")
        result2 = manager.validate_order(order2, state)
        assert result2.is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
