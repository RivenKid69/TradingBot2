# -*- coding: utf-8 -*-
"""
services/portfolio_constraints.py
Portfolio-Level Risk Constraints for Multi-Asset Trading.

This module provides portfolio-level risk management including:
1. PositionLimits - Per-symbol position weight limits
2. SectorExposure - Sector-level exposure limits (GICS-based)
3. FactorTiltLimit - Factor exposure limits (beta, momentum, etc.)
4. validate_order - Pre-trade order validation
5. rebalance_weights - Portfolio rebalancing toward target weights

Architecture:
    PortfolioConstraintManager - Central constraint orchestrator
    └── PositionLimitsValidator
    └── SectorExposureValidator
    └── FactorTiltValidator
    └── RebalanceEngine

Usage:
    from services.portfolio_constraints import (
        PortfolioConstraintManager,
        PositionLimit,
        SectorExposure,
        FactorTiltLimit,
        PortfolioState,
    )

    # Create constraint manager
    manager = PortfolioConstraintManager()
    manager.add_position_limit(PositionLimit("AAPL", max_weight=0.10))
    manager.add_sector_exposure(SectorExposure("TECHNOLOGY", max_weight=0.30))

    # Validate order
    order = Order(symbol="AAPL", side="BUY", qty=100)
    validation = manager.validate_order(order, portfolio_state)

    # Rebalance
    new_weights = manager.rebalance_weights(current_weights)

References:
- GICS Sector Classification: https://www.spglobal.com/spdji/en/landing/investment-themes/gics/
- Risk Parity: Roncalli (2013) "Introduction to Risk Parity"
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Default limits (can be overridden)
DEFAULT_MAX_POSITION_WEIGHT = 0.10  # 10% max per position
DEFAULT_MAX_SECTOR_WEIGHT = 0.30  # 30% max per sector
DEFAULT_MAX_FACTOR_TILT = 2.0  # Max 2x factor exposure

# Tolerance for floating point comparisons
WEIGHT_TOLERANCE = 1e-8

# Factor definitions
FACTOR_BETA = "beta"
FACTOR_MOMENTUM = "momentum"
FACTOR_VALUE = "value"
FACTOR_SIZE = "size"
FACTOR_QUALITY = "quality"
FACTOR_VOLATILITY = "volatility"

# Sector to ETF mapping (GICS-based)
SECTOR_ETFS = {
    "TECHNOLOGY": "XLK",
    "FINANCIALS": "XLF",
    "HEALTHCARE": "XLV",
    "CONSUMER_DISCRETIONARY": "XLY",
    "CONSUMER_STAPLES": "XLP",
    "ENERGY": "XLE",
    "INDUSTRIALS": "XLI",
    "MATERIALS": "XLB",
    "UTILITIES": "XLU",
    "REAL_ESTATE": "XLRE",
    "COMMUNICATION_SERVICES": "XLC",
}

# Symbol to sector mapping (can be extended or loaded from external source)
# This is a partial mapping for common symbols
SYMBOL_TO_SECTOR: Dict[str, str] = {
    # Technology
    "AAPL": "TECHNOLOGY",
    "MSFT": "TECHNOLOGY",
    "GOOGL": "TECHNOLOGY",
    "GOOG": "TECHNOLOGY",
    "AMZN": "CONSUMER_DISCRETIONARY",  # Primarily retail
    "META": "COMMUNICATION_SERVICES",
    "NVDA": "TECHNOLOGY",
    "TSLA": "CONSUMER_DISCRETIONARY",
    "AMD": "TECHNOLOGY",
    "INTC": "TECHNOLOGY",
    # Financials
    "JPM": "FINANCIALS",
    "BAC": "FINANCIALS",
    "WFC": "FINANCIALS",
    "GS": "FINANCIALS",
    "MS": "FINANCIALS",
    # Healthcare
    "JNJ": "HEALTHCARE",
    "UNH": "HEALTHCARE",
    "PFE": "HEALTHCARE",
    "ABBV": "HEALTHCARE",
    "MRK": "HEALTHCARE",
    # Consumer
    "WMT": "CONSUMER_STAPLES",
    "KO": "CONSUMER_STAPLES",
    "PEP": "CONSUMER_STAPLES",
    "PG": "CONSUMER_STAPLES",
    # Energy
    "XOM": "ENERGY",
    "CVX": "ENERGY",
    # Industrials
    "BA": "INDUSTRIALS",
    "CAT": "INDUSTRIALS",
    "GE": "INDUSTRIALS",
    # ETFs (precious metals)
    "GLD": "COMMODITIES",
    "IAU": "COMMODITIES",
    "SLV": "COMMODITIES",
    "SGOL": "COMMODITIES",
    # Index ETFs
    "SPY": "INDEX",
    "QQQ": "INDEX",
    "IWM": "INDEX",
}


# =============================================================================
# Enumerations
# =============================================================================


class ConstraintViolationType(str, Enum):
    """Types of constraint violations."""

    NONE = "NONE"
    POSITION_LIMIT_MAX = "POSITION_LIMIT_MAX"
    POSITION_LIMIT_MIN = "POSITION_LIMIT_MIN"
    SECTOR_LIMIT_MAX = "SECTOR_LIMIT_MAX"
    SECTOR_LIMIT_MIN = "SECTOR_LIMIT_MIN"
    FACTOR_TILT_MAX = "FACTOR_TILT_MAX"
    FACTOR_TILT_MIN = "FACTOR_TILT_MIN"
    TOTAL_EXPOSURE = "TOTAL_EXPOSURE"
    CONCENTRATION = "CONCENTRATION"


class RebalanceAction(str, Enum):
    """Actions from rebalancing."""

    NO_ACTION = "NO_ACTION"
    REDUCE = "REDUCE"
    INCREASE = "INCREASE"
    CLOSE = "CLOSE"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PositionLimit:
    """
    Per-symbol position weight limits.

    Attributes:
        symbol: Trading symbol
        max_weight: Maximum portfolio weight (0.0 to 1.0)
        min_weight: Minimum portfolio weight (-1.0 to max_weight, negative for shorts)
        reason: Optional reason for the limit
    """

    symbol: str
    max_weight: float = DEFAULT_MAX_POSITION_WEIGHT
    min_weight: float = 0.0
    reason: str = ""

    def __post_init__(self) -> None:
        if self.max_weight < 0 or self.max_weight > 1.0:
            raise ValueError(f"max_weight must be in [0, 1], got {self.max_weight}")
        # min_weight can be negative for short positions
        if self.min_weight < -1.0 or self.min_weight > self.max_weight:
            raise ValueError(f"min_weight must be in [-1, max_weight], got {self.min_weight}")

    def validate_weight(self, weight: float) -> Optional[ConstraintViolationType]:
        """Check if weight violates this limit."""
        if weight > self.max_weight + WEIGHT_TOLERANCE:
            return ConstraintViolationType.POSITION_LIMIT_MAX
        if weight < self.min_weight - WEIGHT_TOLERANCE:
            return ConstraintViolationType.POSITION_LIMIT_MIN
        return None


@dataclass
class SectorExposure:
    """
    Sector-level exposure limits.

    Attributes:
        sector: Sector name (e.g., "TECHNOLOGY", "FINANCIALS")
        max_weight: Maximum sector weight (0.0 to 1.0)
        min_weight: Minimum sector weight (0.0 to max_weight)
        reason: Optional reason for the limit
    """

    sector: str
    max_weight: float = DEFAULT_MAX_SECTOR_WEIGHT
    min_weight: float = 0.0
    reason: str = ""

    def __post_init__(self) -> None:
        if self.max_weight < 0 or self.max_weight > 1.0:
            raise ValueError(f"max_weight must be in [0, 1], got {self.max_weight}")
        if self.min_weight < 0 or self.min_weight > self.max_weight:
            raise ValueError(f"min_weight must be in [0, max_weight], got {self.min_weight}")

    def validate_weight(self, weight: float) -> Optional[ConstraintViolationType]:
        """Check if sector weight violates this limit."""
        if weight > self.max_weight + WEIGHT_TOLERANCE:
            return ConstraintViolationType.SECTOR_LIMIT_MAX
        if weight < self.min_weight - WEIGHT_TOLERANCE:
            return ConstraintViolationType.SECTOR_LIMIT_MIN
        return None


@dataclass
class FactorTiltLimit:
    """
    Factor exposure (tilt) limits.

    Attributes:
        factor: Factor name (e.g., "beta", "momentum", "value")
        max_tilt: Maximum absolute factor exposure
        min_tilt: Minimum absolute factor exposure (for mandatory tilts)
        reason: Optional reason for the limit
    """

    factor: str
    max_tilt: float = DEFAULT_MAX_FACTOR_TILT
    min_tilt: float = -DEFAULT_MAX_FACTOR_TILT  # Can be negative for short exposure
    reason: str = ""

    def __post_init__(self) -> None:
        if self.max_tilt < self.min_tilt:
            raise ValueError(f"max_tilt ({self.max_tilt}) must be >= min_tilt ({self.min_tilt})")

    def validate_tilt(self, tilt: float) -> Optional[ConstraintViolationType]:
        """Check if factor tilt violates this limit."""
        if tilt > self.max_tilt + WEIGHT_TOLERANCE:
            return ConstraintViolationType.FACTOR_TILT_MAX
        if tilt < self.min_tilt - WEIGHT_TOLERANCE:
            return ConstraintViolationType.FACTOR_TILT_MIN
        return None


@dataclass
class PortfolioState:
    """
    Current state of the portfolio.

    Attributes:
        positions: Dict of symbol -> (quantity, current_price)
        cash: Available cash
        total_nav: Total net asset value (positions + cash)
        weights: Dict of symbol -> weight (can be computed)
        sector_weights: Dict of sector -> weight
        factor_exposures: Dict of factor -> exposure value
    """

    positions: Dict[str, Tuple[float, float]] = field(
        default_factory=dict
    )  # symbol -> (qty, price)
    cash: float = 0.0
    total_nav: float = 0.0
    weights: Dict[str, float] = field(default_factory=dict)
    sector_weights: Dict[str, float] = field(default_factory=dict)
    factor_exposures: Dict[str, float] = field(default_factory=dict)

    def compute_weights(self) -> None:
        """Compute position weights from positions and total NAV."""
        if self.total_nav <= 0:
            # Compute NAV from positions + cash
            positions_value = sum(qty * price for qty, price in self.positions.values())
            self.total_nav = positions_value + self.cash

        if self.total_nav <= 0:
            self.weights = {}
            return

        self.weights = {}
        for symbol, (qty, price) in self.positions.items():
            position_value = qty * price
            self.weights[symbol] = position_value / self.total_nav

    def compute_sector_weights(
        self,
        symbol_to_sector: Optional[Dict[str, str]] = None,
    ) -> None:
        """Compute sector weights from position weights."""
        if symbol_to_sector is None:
            symbol_to_sector = SYMBOL_TO_SECTOR

        self.sector_weights = {}
        for symbol, weight in self.weights.items():
            sector = symbol_to_sector.get(symbol.upper(), "UNKNOWN")
            if sector not in self.sector_weights:
                self.sector_weights[sector] = 0.0
            self.sector_weights[sector] += weight


@dataclass
class Order:
    """Simple order representation for validation."""

    symbol: str
    side: str  # "BUY" or "SELL"
    qty: float
    price: Optional[float] = None  # Optional limit price
    order_type: str = "MARKET"


@dataclass
class ValidationResult:
    """Result of order validation against constraints."""

    is_valid: bool
    violations: List[ConstraintViolationType] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    adjusted_qty: Optional[float] = None  # Suggested adjusted quantity
    rejection_reason: str = ""

    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0


@dataclass
class RebalanceResult:
    """Result of portfolio rebalancing."""

    target_weights: Dict[str, float]
    actions: Dict[str, Tuple[RebalanceAction, float]]  # symbol -> (action, delta_weight)
    estimated_turnover: float
    messages: List[str] = field(default_factory=list)


# =============================================================================
# Validators
# =============================================================================


class PositionLimitsValidator:
    """Validates position-level weight limits."""

    def __init__(self) -> None:
        self._limits: Dict[str, PositionLimit] = {}
        self._default_limit = PositionLimit("__DEFAULT__", max_weight=1.0, min_weight=0.0)

    def add_limit(self, limit: PositionLimit) -> None:
        """Add a position limit."""
        self._limits[limit.symbol.upper()] = limit

    def remove_limit(self, symbol: str) -> None:
        """Remove a position limit."""
        self._limits.pop(symbol.upper(), None)

    def get_limit(self, symbol: str) -> PositionLimit:
        """Get limit for symbol, or default if not specified."""
        return self._limits.get(symbol.upper(), self._default_limit)

    def set_default_limit(self, limit: PositionLimit) -> None:
        """Set the default limit for symbols without specific limits."""
        self._default_limit = limit

    def validate(
        self,
        weights: Dict[str, float],
    ) -> List[Tuple[str, ConstraintViolationType]]:
        """Validate all position weights. Returns list of (symbol, violation) tuples."""
        violations = []
        for symbol, weight in weights.items():
            limit = self.get_limit(symbol)
            violation = limit.validate_weight(weight)
            if violation:
                violations.append((symbol, violation))
        return violations


class SectorExposureValidator:
    """Validates sector-level exposure limits."""

    def __init__(self) -> None:
        self._limits: Dict[str, SectorExposure] = {}
        self._symbol_to_sector: Dict[str, str] = SYMBOL_TO_SECTOR.copy()

    def add_limit(self, limit: SectorExposure) -> None:
        """Add a sector exposure limit."""
        self._limits[limit.sector.upper()] = limit

    def remove_limit(self, sector: str) -> None:
        """Remove a sector exposure limit."""
        self._limits.pop(sector.upper(), None)

    def get_limit(self, sector: str) -> Optional[SectorExposure]:
        """Get limit for sector, or None if not specified."""
        return self._limits.get(sector.upper())

    def set_symbol_sector(self, symbol: str, sector: str) -> None:
        """Set sector mapping for a symbol."""
        self._symbol_to_sector[symbol.upper()] = sector.upper()

    def get_symbol_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        return self._symbol_to_sector.get(symbol.upper(), "UNKNOWN")

    def compute_sector_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Compute sector weights from position weights."""
        sector_weights: Dict[str, float] = {}
        for symbol, weight in weights.items():
            sector = self.get_symbol_sector(symbol)
            if sector not in sector_weights:
                sector_weights[sector] = 0.0
            sector_weights[sector] += weight
        return sector_weights

    def validate(
        self,
        weights: Dict[str, float],
    ) -> List[Tuple[str, ConstraintViolationType]]:
        """Validate sector weights. Returns list of (sector, violation) tuples."""
        sector_weights = self.compute_sector_weights(weights)
        violations = []

        for sector, weight in sector_weights.items():
            limit = self.get_limit(sector)
            if limit:
                violation = limit.validate_weight(weight)
                if violation:
                    violations.append((sector, violation))

        return violations


class FactorTiltValidator:
    """Validates factor exposure (tilt) limits."""

    def __init__(self) -> None:
        self._limits: Dict[str, FactorTiltLimit] = {}
        self._factor_loadings: Dict[str, Dict[str, float]] = {}  # symbol -> {factor: loading}

    def add_limit(self, limit: FactorTiltLimit) -> None:
        """Add a factor tilt limit."""
        self._limits[limit.factor.lower()] = limit

    def remove_limit(self, factor: str) -> None:
        """Remove a factor tilt limit."""
        self._limits.pop(factor.lower(), None)

    def get_limit(self, factor: str) -> Optional[FactorTiltLimit]:
        """Get limit for factor, or None if not specified."""
        return self._limits.get(factor.lower())

    def set_factor_loadings(self, symbol: str, loadings: Dict[str, float]) -> None:
        """
        Set factor loadings for a symbol.

        Args:
            symbol: Trading symbol
            loadings: Dict of factor -> loading (e.g., {"beta": 1.2, "momentum": 0.5})
        """
        self._factor_loadings[symbol.upper()] = {k.lower(): v for k, v in loadings.items()}

    def get_factor_loading(self, symbol: str, factor: str) -> float:
        """Get factor loading for a symbol. Returns 0.0 if not found."""
        loadings = self._factor_loadings.get(symbol.upper(), {})
        return loadings.get(factor.lower(), 0.0)

    def compute_portfolio_factor_exposure(
        self,
        weights: Dict[str, float],
        factor: str,
    ) -> float:
        """
        Compute weighted portfolio exposure to a factor.

        Returns weighted average of factor loadings.
        """
        total_exposure = 0.0
        for symbol, weight in weights.items():
            loading = self.get_factor_loading(symbol, factor)
            total_exposure += weight * loading
        return total_exposure

    def validate(
        self,
        weights: Dict[str, float],
    ) -> List[Tuple[str, ConstraintViolationType]]:
        """Validate factor exposures. Returns list of (factor, violation) tuples."""
        violations = []

        for factor, limit in self._limits.items():
            exposure = self.compute_portfolio_factor_exposure(weights, factor)
            violation = limit.validate_tilt(exposure)
            if violation:
                violations.append((factor, violation))

        return violations


# =============================================================================
# Rebalance Engine
# =============================================================================


class RebalanceEngine:
    """
    Portfolio rebalancing engine.

    Computes target weights and actions to rebalance portfolio.
    """

    def __init__(
        self,
        position_validator: Optional[PositionLimitsValidator] = None,
        sector_validator: Optional[SectorExposureValidator] = None,
    ) -> None:
        self._position_validator = position_validator
        self._sector_validator = sector_validator

    def rebalance_to_equal_weight(
        self,
        symbols: List[str],
        current_weights: Dict[str, float],
    ) -> RebalanceResult:
        """
        Rebalance to equal weight across symbols.

        Args:
            symbols: List of symbols to include
            current_weights: Current weight per symbol

        Returns:
            RebalanceResult with target weights and actions
        """
        n = len(symbols)
        if n == 0:
            return RebalanceResult(
                target_weights={},
                actions={},
                estimated_turnover=0.0,
            )

        target_weight = 1.0 / n
        target_weights = {s: target_weight for s in symbols}
        return self._compute_rebalance_actions(current_weights, target_weights)

    def rebalance_to_target(
        self,
        target_weights: Dict[str, float],
        current_weights: Dict[str, float],
    ) -> RebalanceResult:
        """
        Rebalance to specified target weights.

        Args:
            target_weights: Desired weight per symbol
            current_weights: Current weight per symbol

        Returns:
            RebalanceResult with target weights and actions
        """
        return self._compute_rebalance_actions(current_weights, target_weights)

    def rebalance_weights(
        self,
        current_weights: Dict[str, float],
        target_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Main rebalancing function - returns adjusted weights.

        If target_weights is None, returns current weights clipped to limits.

        Args:
            current_weights: Current weight per symbol
            target_weights: Optional target weights

        Returns:
            Adjusted weights dict
        """
        if target_weights is None:
            # Just enforce limits on current weights
            return self._enforce_limits(current_weights)

        # Compute rebalance and return target
        result = self.rebalance_to_target(target_weights, current_weights)
        return result.target_weights

    def _enforce_limits(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Enforce position limits on weights, redistribute excess."""
        if self._position_validator is None:
            return weights.copy()

        adjusted = weights.copy()
        total_excess = 0.0
        non_capped_symbols = []

        # First pass: cap at max weights
        for symbol, weight in adjusted.items():
            limit = self._position_validator.get_limit(symbol)
            if weight > limit.max_weight:
                total_excess += weight - limit.max_weight
                adjusted[symbol] = limit.max_weight
            elif weight < limit.min_weight:
                # Force minimum
                adjusted[symbol] = limit.min_weight
            else:
                non_capped_symbols.append(symbol)

        # Redistribute excess to non-capped symbols
        if total_excess > 0 and non_capped_symbols:
            per_symbol_add = total_excess / len(non_capped_symbols)
            for symbol in non_capped_symbols:
                limit = self._position_validator.get_limit(symbol)
                max_add = limit.max_weight - adjusted[symbol]
                adjusted[symbol] += min(per_symbol_add, max_add)

        return adjusted

    def _compute_rebalance_actions(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> RebalanceResult:
        """Compute rebalance actions from current to target weights."""
        actions: Dict[str, Tuple[RebalanceAction, float]] = {}
        messages = []
        total_turnover = 0.0

        # Get all symbols
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())

        for symbol in all_symbols:
            current = current_weights.get(symbol, 0.0)
            target = target_weights.get(symbol, 0.0)
            delta = target - current

            if abs(delta) < WEIGHT_TOLERANCE:
                actions[symbol] = (RebalanceAction.NO_ACTION, 0.0)
            elif delta > 0:
                actions[symbol] = (RebalanceAction.INCREASE, delta)
                total_turnover += delta
            elif target == 0:
                actions[symbol] = (RebalanceAction.CLOSE, -current)
                total_turnover += current
            else:
                actions[symbol] = (RebalanceAction.REDUCE, delta)
                total_turnover += abs(delta)

        return RebalanceResult(
            target_weights=target_weights,
            actions=actions,
            estimated_turnover=total_turnover / 2,  # Turnover is half of total trades
            messages=messages,
        )


# =============================================================================
# Main Constraint Manager
# =============================================================================


class PortfolioConstraintManager:
    """
    Central portfolio constraint manager.

    Orchestrates all constraint validators and provides unified interface
    for order validation and portfolio rebalancing.
    """

    def __init__(
        self,
        position_validator: Optional[PositionLimitsValidator] = None,
        sector_validator: Optional[SectorExposureValidator] = None,
        factor_validator: Optional[FactorTiltValidator] = None,
    ) -> None:
        self._position_validator = position_validator or PositionLimitsValidator()
        self._sector_validator = sector_validator or SectorExposureValidator()
        self._factor_validator = factor_validator or FactorTiltValidator()
        self._rebalance_engine = RebalanceEngine(
            position_validator=self._position_validator,
            sector_validator=self._sector_validator,
        )

    # -------------------------------------------------------------------------
    # Configuration Methods
    # -------------------------------------------------------------------------

    def add_position_limit(self, limit: PositionLimit) -> None:
        """Add a position limit."""
        self._position_validator.add_limit(limit)

    def add_sector_exposure(self, limit: SectorExposure) -> None:
        """Add a sector exposure limit."""
        self._sector_validator.add_limit(limit)

    def add_factor_tilt_limit(self, limit: FactorTiltLimit) -> None:
        """Add a factor tilt limit."""
        self._factor_validator.add_limit(limit)

    def set_symbol_sector(self, symbol: str, sector: str) -> None:
        """Set sector mapping for a symbol."""
        self._sector_validator.set_symbol_sector(symbol, sector)

    def set_factor_loadings(self, symbol: str, loadings: Dict[str, float]) -> None:
        """Set factor loadings for a symbol."""
        self._factor_validator.set_factor_loadings(symbol, loadings)

    def set_default_position_limit(self, limit: PositionLimit) -> None:
        """Set default position limit for symbols without specific limits."""
        self._position_validator.set_default_limit(limit)

    # -------------------------------------------------------------------------
    # Validation Methods
    # -------------------------------------------------------------------------

    def validate_order(
        self,
        order: Order,
        portfolio_state: PortfolioState,
        price: Optional[float] = None,
    ) -> ValidationResult:
        """
        Validate an order against portfolio constraints.

        Args:
            order: Order to validate
            portfolio_state: Current portfolio state
            price: Optional price (uses order.price or last price from state)

        Returns:
            ValidationResult with is_valid, violations, and messages
        """
        violations = []
        messages = []

        # Get price for calculations
        trade_price = price or order.price
        if trade_price is None:
            # Try to get from portfolio state
            if order.symbol in portfolio_state.positions:
                _, trade_price = portfolio_state.positions[order.symbol]
            else:
                return ValidationResult(
                    is_valid=False,
                    violations=[],
                    messages=["Cannot validate: price not available"],
                    rejection_reason="PRICE_NOT_AVAILABLE",
                )

        # Compute post-trade weights
        current_weights = portfolio_state.weights.copy()
        if not current_weights:
            portfolio_state.compute_weights()
            current_weights = portfolio_state.weights.copy()

        # Calculate weight change from order
        order_value = order.qty * trade_price
        if portfolio_state.total_nav <= 0:
            return ValidationResult(
                is_valid=False,
                violations=[],
                messages=["Cannot validate: portfolio NAV is zero or negative"],
                rejection_reason="INVALID_NAV",
            )

        weight_change = order_value / portfolio_state.total_nav
        if order.side.upper() == "SELL":
            weight_change = -weight_change

        # Compute post-trade weight
        current_weight = current_weights.get(order.symbol.upper(), 0.0)
        post_trade_weight = current_weight + weight_change

        # Check for negative position (short)
        if post_trade_weight < -WEIGHT_TOLERANCE:
            # Allow shorts only if min_weight is negative
            limit = self._position_validator.get_limit(order.symbol)
            if limit.min_weight >= 0:
                violations.append(ConstraintViolationType.POSITION_LIMIT_MIN)
                messages.append(f"Order would result in short position for {order.symbol}")

        # Compute hypothetical post-trade weights
        post_trade_weights = current_weights.copy()
        post_trade_weights[order.symbol.upper()] = post_trade_weight

        # Validate position limits
        position_violations = self._position_validator.validate(post_trade_weights)
        for symbol, violation in position_violations:
            violations.append(violation)
            limit = self._position_validator.get_limit(symbol)
            messages.append(
                f"{symbol}: weight {post_trade_weights.get(symbol, 0.0):.2%} "
                f"violates limit [{limit.min_weight:.2%}, {limit.max_weight:.2%}]"
            )

        # Validate sector exposures
        sector_violations = self._sector_validator.validate(post_trade_weights)
        for sector, violation in sector_violations:
            violations.append(violation)
            limit = self._sector_validator.get_limit(sector)
            if limit:
                sector_weight = self._sector_validator.compute_sector_weights(
                    post_trade_weights
                ).get(sector, 0.0)
                messages.append(
                    f"Sector {sector}: weight {sector_weight:.2%} "
                    f"violates limit [{limit.min_weight:.2%}, {limit.max_weight:.2%}]"
                )

        # Validate factor tilts
        factor_violations = self._factor_validator.validate(post_trade_weights)
        for factor, violation in factor_violations:
            violations.append(violation)
            limit = self._factor_validator.get_limit(factor)
            if limit:
                exposure = self._factor_validator.compute_portfolio_factor_exposure(
                    post_trade_weights, factor
                )
                messages.append(
                    f"Factor {factor}: exposure {exposure:.2f} "
                    f"violates limit [{limit.min_tilt:.2f}, {limit.max_tilt:.2f}]"
                )

        is_valid = len(violations) == 0

        # Compute adjusted quantity if violated
        adjusted_qty = None
        if not is_valid and len(position_violations) > 0:
            # Try to find valid quantity
            limit = self._position_validator.get_limit(order.symbol)
            if order.side.upper() == "BUY":
                max_weight = limit.max_weight - current_weight
                max_value = max_weight * portfolio_state.total_nav
                adjusted_qty = max(0.0, max_value / trade_price)
            else:
                min_weight = current_weight - limit.min_weight
                max_value = min_weight * portfolio_state.total_nav
                adjusted_qty = max(0.0, max_value / trade_price)

        return ValidationResult(
            is_valid=is_valid,
            violations=violations,
            messages=messages,
            adjusted_qty=adjusted_qty,
            rejection_reason="" if is_valid else violations[0].value,
        )

    def validate_portfolio_state(
        self,
        portfolio_state: PortfolioState,
    ) -> ValidationResult:
        """
        Validate entire portfolio state against constraints.

        Args:
            portfolio_state: Portfolio state to validate

        Returns:
            ValidationResult with all violations
        """
        violations = []
        messages = []

        # Ensure weights are computed
        if not portfolio_state.weights:
            portfolio_state.compute_weights()

        weights = portfolio_state.weights

        # Validate position limits
        position_violations = self._position_validator.validate(weights)
        for symbol, violation in position_violations:
            violations.append(violation)
            messages.append(f"Position {symbol} violates limit")

        # Validate sector exposures
        sector_violations = self._sector_validator.validate(weights)
        for sector, violation in sector_violations:
            violations.append(violation)
            messages.append(f"Sector {sector} violates limit")

        # Validate factor tilts
        factor_violations = self._factor_validator.validate(weights)
        for factor, violation in factor_violations:
            violations.append(violation)
            messages.append(f"Factor {factor} violates limit")

        return ValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            messages=messages,
        )

    # -------------------------------------------------------------------------
    # Rebalancing Methods
    # -------------------------------------------------------------------------

    def rebalance_weights(
        self,
        current_weights: Dict[str, float],
        target_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Rebalance portfolio weights.

        If target_weights is None, enforces current limits.

        Args:
            current_weights: Current weight per symbol
            target_weights: Optional target weights

        Returns:
            Adjusted weights dict respecting all constraints
        """
        return self._rebalance_engine.rebalance_weights(current_weights, target_weights)

    def rebalance_to_equal_weight(
        self,
        symbols: List[str],
        current_weights: Dict[str, float],
    ) -> RebalanceResult:
        """
        Rebalance to equal weight across specified symbols.

        Args:
            symbols: Symbols to include in equal weight portfolio
            current_weights: Current weights

        Returns:
            RebalanceResult with actions
        """
        return self._rebalance_engine.rebalance_to_equal_weight(symbols, current_weights)

    def get_rebalance_actions(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> RebalanceResult:
        """
        Get detailed rebalance actions.

        Args:
            current_weights: Current weights
            target_weights: Target weights

        Returns:
            RebalanceResult with per-symbol actions
        """
        return self._rebalance_engine.rebalance_to_target(target_weights, current_weights)

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    def get_sector_weights(
        self,
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Get sector-level weights from position weights."""
        return self._sector_validator.compute_sector_weights(weights)

    def get_factor_exposure(
        self,
        weights: Dict[str, float],
        factor: str,
    ) -> float:
        """Get portfolio exposure to a factor."""
        return self._factor_validator.compute_portfolio_factor_exposure(weights, factor)


# =============================================================================
# Factory Functions
# =============================================================================


def create_constraint_manager(
    max_position_weight: float = DEFAULT_MAX_POSITION_WEIGHT,
    max_sector_weight: float = DEFAULT_MAX_SECTOR_WEIGHT,
    sectors_to_limit: Optional[List[str]] = None,
) -> PortfolioConstraintManager:
    """
    Create a constraint manager with common defaults.

    Args:
        max_position_weight: Default max weight per position
        max_sector_weight: Max weight per sector
        sectors_to_limit: Optional list of sectors to limit

    Returns:
        Configured PortfolioConstraintManager
    """
    manager = PortfolioConstraintManager()

    # Set default position limit
    manager.set_default_position_limit(PositionLimit("__DEFAULT__", max_weight=max_position_weight))

    # Add sector limits
    sectors = sectors_to_limit or list(SECTOR_ETFS.keys())
    for sector in sectors:
        manager.add_sector_exposure(SectorExposure(sector, max_weight=max_sector_weight))

    return manager


def validate_order(
    order: Order,
    portfolio_state: PortfolioState,
    constraints: Optional[PortfolioConstraintManager] = None,
) -> ValidationResult:
    """
    Convenience function to validate an order.

    Args:
        order: Order to validate
        portfolio_state: Current portfolio state
        constraints: Optional constraint manager (creates default if None)

    Returns:
        ValidationResult
    """
    if constraints is None:
        constraints = create_constraint_manager()
    return constraints.validate_order(order, portfolio_state)


def rebalance_weights(
    current: Dict[str, float],
    target: Optional[Dict[str, float]] = None,
    constraints: Optional[PortfolioConstraintManager] = None,
) -> Dict[str, float]:
    """
    Convenience function to rebalance weights.

    Args:
        current: Current weights
        target: Optional target weights
        constraints: Optional constraint manager (creates default if None)

    Returns:
        Adjusted weights
    """
    if constraints is None:
        constraints = create_constraint_manager()
    return constraints.rebalance_weights(current, target)
