# -*- coding: utf-8 -*-
"""
Market Impact Models for L3 LOB Simulation.

This module implements sophisticated market impact models that decompose
the total impact into temporary (transient) and permanent components.

Models:
    1. KyleLambdaModel: Classic Kyle (1985) linear impact model
       impact = λ * sign(order) * sqrt(volume)

    2. AlmgrenChrissModel: Optimal execution framework
       temp_impact = η * σ * (Q/ADV)^0.5
       perm_impact = γ * (Q/ADV)

    3. GatheralModel: Transient impact with power-law decay
       temp_impact = η * σ * sign(v) * |v|^δ (δ ≈ 0.5)
       perm_impact = γ * v
       decay = G(t) = (1 + t/τ)^(-β)

References:
    - Kyle (1985): "Continuous Auctions and Insider Trading"
    - Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
    - Gatheral (2010): "No-Dynamic-Arbitrage and Market Impact"
    - Obizhaeva & Wang (2013): "Optimal Trading Strategy and Supply/Demand Dynamics"

Performance Target: <10μs per impact computation
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from lob.data_structures import (
    LimitOrder,
    OrderBook,
    Side,
)


# ==============================================================================
# Enums and Constants
# ==============================================================================


class ImpactModelType(IntEnum):
    """Market impact model type enumeration."""

    KYLE_LAMBDA = 1  # Kyle (1985) linear
    ALMGREN_CHRISS = 2  # Almgren-Chriss (2001) square-root
    GATHERAL = 3  # Gatheral (2010) transient with decay
    OBIZHAEVA_WANG = 4  # Obizhaeva-Wang (2013) resilience
    LINEAR = 5  # Simple linear model
    COMPOSITE = 6  # Weighted combination


class DecayType(IntEnum):
    """Impact decay function type."""

    EXPONENTIAL = 1  # exp(-t/τ)
    POWER_LAW = 2  # (1 + t/τ)^(-β)
    LINEAR = 3  # max(0, 1 - t/τ)
    NONE = 4  # No decay (permanent only)


# Default constants (calibrated for US equities)
_DEFAULT_IMPACT_COEF_TEMP = 0.1  # η: temporary impact coefficient
_DEFAULT_IMPACT_COEF_PERM = 0.05  # γ: permanent impact coefficient
_DEFAULT_IMPACT_EXPONENT = 0.5  # δ: impact exponent (square-root)
_DEFAULT_DECAY_HALF_LIFE_MS = 60000  # τ: 60 second half-life
_DEFAULT_DECAY_BETA = 1.5  # β: power-law decay exponent
_DEFAULT_VOLATILITY = 0.02  # σ: 2% daily volatility
_MIN_ADV = 1.0  # Minimum ADV to prevent division by zero


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass
class ImpactParameters:
    """
    Parameters for market impact models.

    Attributes:
        eta: Temporary impact coefficient (η)
        gamma: Permanent impact coefficient (γ)
        delta: Impact exponent (δ), typically 0.5 for square-root
        tau_ms: Decay time constant in milliseconds (τ)
        beta: Power-law decay exponent (β)
        volatility: Daily volatility (σ)
        spread_bps: Typical spread in basis points
    """

    eta: float = _DEFAULT_IMPACT_COEF_TEMP  # Temporary impact
    gamma: float = _DEFAULT_IMPACT_COEF_PERM  # Permanent impact
    delta: float = _DEFAULT_IMPACT_EXPONENT  # Impact exponent
    tau_ms: float = _DEFAULT_DECAY_HALF_LIFE_MS  # Decay time constant
    beta: float = _DEFAULT_DECAY_BETA  # Power-law exponent
    volatility: float = _DEFAULT_VOLATILITY  # Daily volatility
    spread_bps: float = 5.0  # Typical spread

    @classmethod
    def for_equity(cls) -> "ImpactParameters":
        """Default parameters for US equities."""
        return cls(
            eta=0.05,  # Lower impact for liquid equities
            gamma=0.03,
            delta=0.5,
            tau_ms=30000,  # 30s decay
            beta=1.5,
            volatility=0.02,
            spread_bps=2.0,
        )

    @classmethod
    def for_crypto(cls) -> "ImpactParameters":
        """Default parameters for crypto markets."""
        return cls(
            eta=0.10,  # Higher impact
            gamma=0.05,
            delta=0.5,
            tau_ms=60000,  # 60s decay
            beta=1.2,
            volatility=0.05,
            spread_bps=5.0,
        )


@dataclass
class ImpactResult:
    """
    Result of market impact computation.

    Attributes:
        temporary_impact_bps: Temporary impact in basis points (reverts)
        permanent_impact_bps: Permanent impact in basis points (doesn't revert)
        total_impact_bps: Total impact (temp + perm)
        impact_cost: Estimated impact cost in quote currency
        price_adjustment: Expected price adjustment from mid
        decay_profile: Impact decay at future time points [(t_ms, impact_bps), ...]
        model_type: Model used for computation
        details: Additional model-specific details
    """

    temporary_impact_bps: float = 0.0
    permanent_impact_bps: float = 0.0
    total_impact_bps: float = 0.0
    impact_cost: float = 0.0
    price_adjustment: float = 0.0  # In price units
    decay_profile: List[Tuple[int, float]] = field(default_factory=list)
    model_type: ImpactModelType = ImpactModelType.ALMGREN_CHRISS
    details: Dict[str, float] = field(default_factory=dict)

    @property
    def temporary_fraction(self) -> float:
        """Fraction of impact that is temporary."""
        if self.total_impact_bps <= 0:
            return 0.0
        return self.temporary_impact_bps / self.total_impact_bps

    @property
    def permanent_fraction(self) -> float:
        """Fraction of impact that is permanent."""
        if self.total_impact_bps <= 0:
            return 0.0
        return self.permanent_impact_bps / self.total_impact_bps


@dataclass
class ImpactState:
    """
    Tracks cumulative market impact state over time.

    Used for managing impact decay and cumulative effects.

    Attributes:
        cumulative_temp_impact_bps: Current cumulative temporary impact
        cumulative_perm_impact_bps: Current cumulative permanent impact
        last_update_ms: Timestamp of last update
        impact_history: History of impacts [(timestamp_ms, temp_bps, perm_bps), ...]
    """

    cumulative_temp_impact_bps: float = 0.0
    cumulative_perm_impact_bps: float = 0.0
    last_update_ms: int = 0
    impact_history: List[Tuple[int, float, float]] = field(default_factory=list)

    @property
    def total_impact_bps(self) -> float:
        """Current total impact."""
        return self.cumulative_temp_impact_bps + self.cumulative_perm_impact_bps

    def add_impact(
        self,
        timestamp_ms: int,
        temp_impact_bps: float,
        perm_impact_bps: float,
    ) -> None:
        """Add new impact to state."""
        self.cumulative_temp_impact_bps += temp_impact_bps
        self.cumulative_perm_impact_bps += perm_impact_bps
        self.last_update_ms = timestamp_ms
        self.impact_history.append((timestamp_ms, temp_impact_bps, perm_impact_bps))

    def decay_temporary(self, decay_factor: float) -> None:
        """Apply decay to temporary impact component."""
        self.cumulative_temp_impact_bps *= decay_factor

    def reset(self) -> None:
        """Reset impact state."""
        self.cumulative_temp_impact_bps = 0.0
        self.cumulative_perm_impact_bps = 0.0
        self.last_update_ms = 0
        self.impact_history.clear()


# ==============================================================================
# Abstract Base Class
# ==============================================================================


class MarketImpactModel(ABC):
    """
    Abstract base class for market impact models.

    Provides interface for computing temporary and permanent impact
    components, as well as impact decay over time.
    """

    @property
    @abstractmethod
    def model_type(self) -> ImpactModelType:
        """Return model type identifier."""
        pass

    @abstractmethod
    def compute_temporary_impact(
        self,
        order_qty: float,
        adv: float,
        volatility: float,
        side: Optional[Side] = None,
    ) -> float:
        """
        Compute temporary (transient) market impact.

        Temporary impact reverts after the trade as market makers
        restore liquidity and prices mean-revert.

        Args:
            order_qty: Order quantity
            adv: Average daily volume
            volatility: Current volatility estimate
            side: Order side (for directional impact)

        Returns:
            Temporary impact in basis points
        """
        pass

    @abstractmethod
    def compute_permanent_impact(
        self,
        order_qty: float,
        adv: float,
        side: Optional[Side] = None,
    ) -> float:
        """
        Compute permanent market impact.

        Permanent impact doesn't revert - represents information
        revealed by the trade that shifts the fundamental price.

        Args:
            order_qty: Order quantity
            adv: Average daily volume
            side: Order side (for directional impact)

        Returns:
            Permanent impact in basis points
        """
        pass

    @abstractmethod
    def compute_decay(
        self,
        impact_bps: float,
        time_since_trade_ms: int,
    ) -> float:
        """
        Compute impact decay over time.

        Models how temporary impact diminishes as time passes.

        Args:
            impact_bps: Initial impact in basis points
            time_since_trade_ms: Time elapsed since trade in milliseconds

        Returns:
            Remaining impact in basis points
        """
        pass

    def compute_total_impact(
        self,
        order_qty: float,
        adv: float,
        volatility: float,
        mid_price: float,
        side: Optional[Side] = None,
    ) -> ImpactResult:
        """
        Compute total market impact with decomposition.

        Args:
            order_qty: Order quantity
            adv: Average daily volume
            volatility: Current volatility estimate
            mid_price: Current mid-market price
            side: Order side

        Returns:
            ImpactResult with full impact decomposition
        """
        temp_bps = self.compute_temporary_impact(order_qty, adv, volatility, side)
        perm_bps = self.compute_permanent_impact(order_qty, adv, side)
        total_bps = temp_bps + perm_bps

        # Compute price adjustment
        notional = order_qty * mid_price
        impact_cost = notional * total_bps / 10000.0
        price_adj = mid_price * total_bps / 10000.0

        # Generate decay profile (0, 10s, 30s, 60s, 120s, 300s)
        decay_times_ms = [0, 10000, 30000, 60000, 120000, 300000]
        decay_profile = []
        for t_ms in decay_times_ms:
            remaining_temp = self.compute_decay(temp_bps, t_ms)
            total_at_t = remaining_temp + perm_bps
            decay_profile.append((t_ms, total_at_t))

        return ImpactResult(
            temporary_impact_bps=temp_bps,
            permanent_impact_bps=perm_bps,
            total_impact_bps=total_bps,
            impact_cost=impact_cost,
            price_adjustment=price_adj,
            decay_profile=decay_profile,
            model_type=self.model_type,
            details={
                "order_qty": order_qty,
                "adv": adv,
                "volatility": volatility,
                "participation_ratio": order_qty / max(adv, _MIN_ADV),
            },
        )

    def compute_for_order(
        self,
        order: LimitOrder,
        order_book: OrderBook,
        adv: float,
        volatility: float,
    ) -> ImpactResult:
        """
        Compute market impact for a specific order.

        Convenience method that extracts parameters from order/book.
        """
        mid_price = order_book.mid_price or order.price
        return self.compute_total_impact(
            order_qty=order.remaining_qty,
            adv=adv,
            volatility=volatility,
            mid_price=mid_price,
            side=order.side,
        )


# ==============================================================================
# Kyle Lambda Model
# ==============================================================================


class KyleLambdaModel(MarketImpactModel):
    """
    Kyle (1985) Lambda Model for market impact.

    Classic linear price impact model where:
        Δp = λ * sign(x) * |x|

    Where:
        λ = Kyle's lambda (price impact per unit volume)
        x = signed order flow

    Properties:
        - Linear in order size
        - Information-driven impact
        - Used as baseline for more sophisticated models

    Reference:
        Kyle (1985): "Continuous Auctions and Insider Trading"
    """

    def __init__(
        self,
        lambda_coef: float = 0.0001,  # Impact per unit volume
        permanent_fraction: float = 0.5,  # Fraction that is permanent
        decay_type: DecayType = DecayType.EXPONENTIAL,
        decay_tau_ms: float = _DEFAULT_DECAY_HALF_LIFE_MS,
    ) -> None:
        """
        Initialize Kyle Lambda model.

        Args:
            lambda_coef: Kyle's lambda (price impact coefficient)
            permanent_fraction: Fraction of impact that is permanent [0, 1]
            decay_type: Type of decay function for temporary impact
            decay_tau_ms: Decay time constant in milliseconds
        """
        self._lambda = float(lambda_coef)
        self._perm_frac = max(0.0, min(1.0, float(permanent_fraction)))
        self._temp_frac = 1.0 - self._perm_frac
        self._decay_type = decay_type
        self._tau_ms = float(decay_tau_ms)

    @property
    def model_type(self) -> ImpactModelType:
        return ImpactModelType.KYLE_LAMBDA

    def compute_temporary_impact(
        self,
        order_qty: float,
        adv: float,
        volatility: float,
        side: Optional[Side] = None,
    ) -> float:
        """
        Compute temporary impact using Kyle model.

        temp_impact = λ * |Q| * (1 - perm_fraction) * 10000 (bps)
        """
        impact = self._lambda * abs(order_qty) * self._temp_frac
        return impact * 10000.0  # Convert to bps

    def compute_permanent_impact(
        self,
        order_qty: float,
        adv: float,
        side: Optional[Side] = None,
    ) -> float:
        """
        Compute permanent impact using Kyle model.

        perm_impact = λ * |Q| * perm_fraction * 10000 (bps)
        """
        impact = self._lambda * abs(order_qty) * self._perm_frac
        return impact * 10000.0  # Convert to bps

    def compute_decay(
        self,
        impact_bps: float,
        time_since_trade_ms: int,
    ) -> float:
        """Compute impact decay based on decay type."""
        if time_since_trade_ms <= 0:
            return impact_bps

        t = float(time_since_trade_ms)
        tau = self._tau_ms

        if self._decay_type == DecayType.EXPONENTIAL:
            # exp(-t/τ)
            decay = math.exp(-t / tau)
        elif self._decay_type == DecayType.POWER_LAW:
            # (1 + t/τ)^(-β), using β=1.5 by default
            decay = (1.0 + t / tau) ** (-1.5)
        elif self._decay_type == DecayType.LINEAR:
            # max(0, 1 - t/τ)
            decay = max(0.0, 1.0 - t / tau)
        else:  # DecayType.NONE
            decay = 1.0

        return impact_bps * decay


# ==============================================================================
# Almgren-Chriss Model
# ==============================================================================


class AlmgrenChrissModel(MarketImpactModel):
    """
    Almgren-Chriss (2001) Market Impact Model.

    Framework for optimal execution with market impact:
        temp_impact = η * σ * (Q/V)^δ
        perm_impact = γ * (Q/V)

    Where:
        η = temporary impact coefficient
        γ = permanent impact coefficient
        σ = daily volatility
        Q = order quantity
        V = average daily volume (ADV)
        δ = impact exponent (typically 0.5 for square-root)

    Key insights:
        - Temporary impact scales as sqrt(participation) - empirically validated
        - Permanent impact scales linearly with participation
        - Total impact = temp + perm

    Reference:
        Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
    """

    def __init__(
        self,
        params: Optional[ImpactParameters] = None,
        eta: Optional[float] = None,
        gamma: Optional[float] = None,
        delta: float = _DEFAULT_IMPACT_EXPONENT,
        decay_type: DecayType = DecayType.EXPONENTIAL,
    ) -> None:
        """
        Initialize Almgren-Chriss model.

        Args:
            params: ImpactParameters object (overrides eta, gamma)
            eta: Temporary impact coefficient (default: 0.1)
            gamma: Permanent impact coefficient (default: 0.05)
            delta: Impact exponent (default: 0.5 for square-root)
            decay_type: Decay function for temporary impact
        """
        if params is not None:
            self._params = params
        else:
            self._params = ImpactParameters(
                eta=eta if eta is not None else _DEFAULT_IMPACT_COEF_TEMP,
                gamma=gamma if gamma is not None else _DEFAULT_IMPACT_COEF_PERM,
                delta=delta,
            )
        self._decay_type = decay_type

    @property
    def model_type(self) -> ImpactModelType:
        return ImpactModelType.ALMGREN_CHRISS

    @property
    def params(self) -> ImpactParameters:
        """Get model parameters."""
        return self._params

    def compute_temporary_impact(
        self,
        order_qty: float,
        adv: float,
        volatility: float,
        side: Optional[Side] = None,
    ) -> float:
        """
        Compute temporary impact using square-root model.

        temp_impact_bps = η * σ * (Q/V)^δ * 10000
        """
        if adv <= _MIN_ADV:
            return 0.0

        participation = abs(order_qty) / adv
        vol = volatility if volatility > 0 else self._params.volatility

        # Square-root impact (or general power-law with δ)
        impact = self._params.eta * vol * (participation**self._params.delta)

        return impact * 10000.0  # Convert to bps

    def compute_permanent_impact(
        self,
        order_qty: float,
        adv: float,
        side: Optional[Side] = None,
    ) -> float:
        """
        Compute permanent impact (linear in participation).

        perm_impact_bps = γ * (Q/V) * 10000
        """
        if adv <= _MIN_ADV:
            return 0.0

        participation = abs(order_qty) / adv
        impact = self._params.gamma * participation

        return impact * 10000.0  # Convert to bps

    def compute_decay(
        self,
        impact_bps: float,
        time_since_trade_ms: int,
    ) -> float:
        """Compute decay using configured decay type."""
        if time_since_trade_ms <= 0:
            return impact_bps

        t = float(time_since_trade_ms)
        tau = self._params.tau_ms

        if self._decay_type == DecayType.EXPONENTIAL:
            decay = math.exp(-t / tau)
        elif self._decay_type == DecayType.POWER_LAW:
            decay = (1.0 + t / tau) ** (-self._params.beta)
        elif self._decay_type == DecayType.LINEAR:
            decay = max(0.0, 1.0 - t / tau)
        else:
            decay = 1.0

        return impact_bps * decay

    def compute_optimal_execution_time(
        self,
        order_qty: float,
        adv: float,
        volatility: float,
        risk_aversion: float = 1e-6,
    ) -> float:
        """
        Compute optimal execution time using Almgren-Chriss framework.

        T* = sqrt(Q * η / (λ * γ * σ^2))

        Where λ = risk aversion coefficient.

        Args:
            order_qty: Total order quantity
            adv: Average daily volume
            volatility: Daily volatility
            risk_aversion: Risk aversion coefficient (λ)

        Returns:
            Optimal execution time in seconds
        """
        if adv <= _MIN_ADV or volatility <= 0:
            return 0.0

        eta = self._params.eta
        gamma = self._params.gamma
        sigma = volatility

        # Optimal time in ADV units
        if risk_aversion * gamma * (sigma**2) > 0:
            t_star = math.sqrt(abs(order_qty) * eta / (risk_aversion * gamma * (sigma**2)))
            # Convert to seconds (assuming ADV is daily)
            return t_star * 24 * 3600  # Seconds in a trading day equivalent

        return 3600.0  # Default: 1 hour

    def compute_execution_trajectory(
        self,
        order_qty: float,
        execution_time_sec: float,
        n_intervals: int = 10,
    ) -> List[Tuple[float, float]]:
        """
        Compute optimal execution trajectory (TWAP baseline).

        Args:
            order_qty: Total order quantity
            execution_time_sec: Total execution time
            n_intervals: Number of trading intervals

        Returns:
            List of (time_sec, cumulative_qty) tuples
        """
        if n_intervals <= 0:
            return [(0.0, order_qty)]

        trajectory = []
        dt = execution_time_sec / n_intervals
        qty_per_interval = order_qty / n_intervals

        for i in range(n_intervals + 1):
            t = i * dt
            cum_qty = min(i * qty_per_interval, order_qty)
            trajectory.append((t, cum_qty))

        return trajectory


# ==============================================================================
# Gatheral Model
# ==============================================================================


class GatheralModel(MarketImpactModel):
    """
    Gatheral (2010) Transient Impact Model.

    Extended model with power-law decay:
        temp_impact = η * σ * sign(v) * |v|^δ
        perm_impact = γ * v
        decay = G(t) = (1 + t/τ)^(-β)

    Key features:
        - Power-law decay (slower than exponential)
        - Matches empirical observations of impact persistence
        - Propagator kernel captures market resilience

    Reference:
        Gatheral (2010): "No-Dynamic-Arbitrage and Market Impact"
        Gatheral, Schied & Slynko (2012): "Transient Linear Price Impact"
    """

    def __init__(
        self,
        params: Optional[ImpactParameters] = None,
        eta: float = _DEFAULT_IMPACT_COEF_TEMP,
        gamma: float = _DEFAULT_IMPACT_COEF_PERM,
        delta: float = _DEFAULT_IMPACT_EXPONENT,
        tau_ms: float = _DEFAULT_DECAY_HALF_LIFE_MS,
        beta: float = _DEFAULT_DECAY_BETA,
    ) -> None:
        """
        Initialize Gatheral model.

        Args:
            params: ImpactParameters object
            eta: Temporary impact coefficient
            gamma: Permanent impact coefficient
            delta: Impact exponent
            tau_ms: Decay time constant (milliseconds)
            beta: Power-law decay exponent
        """
        if params is not None:
            self._params = params
        else:
            self._params = ImpactParameters(
                eta=eta,
                gamma=gamma,
                delta=delta,
                tau_ms=tau_ms,
                beta=beta,
            )

    @property
    def model_type(self) -> ImpactModelType:
        return ImpactModelType.GATHERAL

    @property
    def params(self) -> ImpactParameters:
        """Get model parameters."""
        return self._params

    def compute_temporary_impact(
        self,
        order_qty: float,
        adv: float,
        volatility: float,
        side: Optional[Side] = None,
    ) -> float:
        """
        Compute temporary impact with signed quantity.

        temp_impact = η * σ * sign(v) * |v|^δ (in price terms)

        Returns impact in basis points (unsigned).
        """
        if adv <= _MIN_ADV:
            return 0.0

        participation = abs(order_qty) / adv
        vol = volatility if volatility > 0 else self._params.volatility

        # Gatheral power-law impact
        impact = self._params.eta * vol * (participation**self._params.delta)

        return impact * 10000.0  # Convert to bps

    def compute_permanent_impact(
        self,
        order_qty: float,
        adv: float,
        side: Optional[Side] = None,
    ) -> float:
        """
        Compute permanent impact (linear component).

        perm_impact = γ * (Q/V)
        """
        if adv <= _MIN_ADV:
            return 0.0

        participation = abs(order_qty) / adv
        impact = self._params.gamma * participation

        return impact * 10000.0  # Convert to bps

    def compute_decay(
        self,
        impact_bps: float,
        time_since_trade_ms: int,
    ) -> float:
        """
        Compute power-law decay (Gatheral propagator kernel).

        G(t) = (1 + t/τ)^(-β)

        Power-law decays slower than exponential, matching
        empirical observations of impact persistence.
        """
        if time_since_trade_ms <= 0:
            return impact_bps

        t = float(time_since_trade_ms)
        tau = self._params.tau_ms
        beta = self._params.beta

        # Power-law decay
        decay = (1.0 + t / tau) ** (-beta)

        return impact_bps * decay

    def compute_resilience_rate(self) -> float:
        """
        Compute market resilience rate (inverse of decay time).

        Higher values = faster recovery from impact.

        Returns:
            Resilience rate (1/ms)
        """
        return 1.0 / self._params.tau_ms if self._params.tau_ms > 0 else 0.0

    def compute_impact_at_time(
        self,
        order_qty: float,
        adv: float,
        volatility: float,
        time_since_trade_ms: int,
    ) -> Tuple[float, float]:
        """
        Compute temporary and permanent impact at a specific time.

        Args:
            order_qty: Order quantity
            adv: Average daily volume
            volatility: Volatility
            time_since_trade_ms: Time since trade

        Returns:
            Tuple of (temporary_impact_bps, permanent_impact_bps)
        """
        initial_temp = self.compute_temporary_impact(order_qty, adv, volatility)
        perm = self.compute_permanent_impact(order_qty, adv)

        # Apply decay to temporary component
        remaining_temp = self.compute_decay(initial_temp, time_since_trade_ms)

        return remaining_temp, perm


# ==============================================================================
# Composite Model
# ==============================================================================


class CompositeImpactModel(MarketImpactModel):
    """
    Composite model combining multiple impact models.

    Allows weighted combination of different models for ensemble estimation.
    """

    def __init__(
        self,
        models: Optional[List[Tuple[MarketImpactModel, float]]] = None,
    ) -> None:
        """
        Initialize composite model.

        Args:
            models: List of (model, weight) tuples
        """
        if models is None:
            # Default: equal weight of Almgren-Chriss and Gatheral
            self._models = [
                (AlmgrenChrissModel(), 0.5),
                (GatheralModel(), 0.5),
            ]
        else:
            self._models = models

        # Normalize weights
        total = sum(w for _, w in self._models)
        if total > 0:
            self._models = [(m, w / total) for m, w in self._models]

    @property
    def model_type(self) -> ImpactModelType:
        return ImpactModelType.COMPOSITE

    def compute_temporary_impact(
        self,
        order_qty: float,
        adv: float,
        volatility: float,
        side: Optional[Side] = None,
    ) -> float:
        """Weighted average of model temporary impacts."""
        total = 0.0
        for model, weight in self._models:
            total += weight * model.compute_temporary_impact(order_qty, adv, volatility, side)
        return total

    def compute_permanent_impact(
        self,
        order_qty: float,
        adv: float,
        side: Optional[Side] = None,
    ) -> float:
        """Weighted average of model permanent impacts."""
        total = 0.0
        for model, weight in self._models:
            total += weight * model.compute_permanent_impact(order_qty, adv, side)
        return total

    def compute_decay(
        self,
        impact_bps: float,
        time_since_trade_ms: int,
    ) -> float:
        """Weighted average of model decays."""
        total = 0.0
        for model, weight in self._models:
            total += weight * model.compute_decay(impact_bps, time_since_trade_ms)
        return total


# ==============================================================================
# Impact Tracker
# ==============================================================================


class ImpactTracker:
    """
    Tracks cumulative market impact over multiple trades.

    Maintains impact state and applies decay over time for
    realistic multi-trade impact simulation.
    """

    def __init__(
        self,
        model: Optional[MarketImpactModel] = None,
        max_history: int = 1000,
    ) -> None:
        """
        Initialize impact tracker.

        Args:
            model: Market impact model to use
            max_history: Maximum number of historical impacts to track
        """
        self._model = model or AlmgrenChrissModel()
        self._max_history = max_history
        self._state = ImpactState()
        self._trades: List[Tuple[int, float, float, float]] = []  # (ts, qty, adv, vol)

    @property
    def model(self) -> MarketImpactModel:
        """Get the impact model."""
        return self._model

    @property
    def state(self) -> ImpactState:
        """Get current impact state."""
        return self._state

    @property
    def current_impact_bps(self) -> float:
        """Get current total impact in basis points."""
        return self._state.total_impact_bps

    def record_trade(
        self,
        timestamp_ms: int,
        order_qty: float,
        adv: float,
        volatility: float,
    ) -> ImpactResult:
        """
        Record a trade and update impact state.

        Args:
            timestamp_ms: Trade timestamp
            order_qty: Trade quantity
            adv: Average daily volume
            volatility: Current volatility

        Returns:
            ImpactResult for this trade
        """
        # Apply decay for time elapsed since last update
        if self._state.last_update_ms > 0:
            elapsed = timestamp_ms - self._state.last_update_ms
            if elapsed > 0:
                decay_factor = self._model.compute_decay(1.0, elapsed)
                self._state.decay_temporary(decay_factor)

        # Compute new impact
        temp_bps = self._model.compute_temporary_impact(order_qty, adv, volatility)
        perm_bps = self._model.compute_permanent_impact(order_qty, adv)

        # Update state
        self._state.add_impact(timestamp_ms, temp_bps, perm_bps)

        # Track trade
        self._trades.append((timestamp_ms, order_qty, adv, volatility))
        if len(self._trades) > self._max_history:
            self._trades.pop(0)

        return ImpactResult(
            temporary_impact_bps=temp_bps,
            permanent_impact_bps=perm_bps,
            total_impact_bps=temp_bps + perm_bps,
            model_type=self._model.model_type,
            details={
                "cumulative_temp_bps": self._state.cumulative_temp_impact_bps,
                "cumulative_perm_bps": self._state.cumulative_perm_impact_bps,
            },
        )

    def get_impact_at_time(self, timestamp_ms: int) -> float:
        """
        Get estimated total impact at a specific timestamp.

        Applies appropriate decay based on time elapsed.
        """
        if timestamp_ms < self._state.last_update_ms:
            # Looking at past - would need full recomputation
            return self._state.total_impact_bps

        # Decay temporary impact for elapsed time
        elapsed = timestamp_ms - self._state.last_update_ms
        if elapsed <= 0:
            return self._state.total_impact_bps

        decayed_temp = self._model.compute_decay(self._state.cumulative_temp_impact_bps, elapsed)
        return decayed_temp + self._state.cumulative_perm_impact_bps

    def reset(self) -> None:
        """Reset tracker state."""
        self._state.reset()
        self._trades.clear()


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_impact_model(
    model_type: Union[str, ImpactModelType] = "almgren_chriss",
    asset_class: str = "equity",
    **kwargs,
) -> MarketImpactModel:
    """
    Factory function to create market impact models.

    Args:
        model_type: Model type name or enum
        asset_class: Asset class ("equity", "crypto")
        **kwargs: Model-specific parameters

    Returns:
        MarketImpactModel instance

    Example:
        model = create_impact_model("gatheral", asset_class="equity")
        model = create_impact_model("almgren_chriss", eta=0.08, gamma=0.04)
    """
    # Get default parameters for asset class
    if asset_class.lower() == "equity":
        default_params = ImpactParameters.for_equity()
    else:
        default_params = ImpactParameters.for_crypto()

    # Override with provided kwargs
    if kwargs:
        param_fields = ["eta", "gamma", "delta", "tau_ms", "beta", "volatility", "spread_bps"]
        param_kwargs = {k: v for k, v in kwargs.items() if k in param_fields}
        for k, v in param_kwargs.items():
            setattr(default_params, k, v)

    # Parse model type
    if isinstance(model_type, str):
        type_map = {
            "kyle": ImpactModelType.KYLE_LAMBDA,
            "kyle_lambda": ImpactModelType.KYLE_LAMBDA,
            "almgren": ImpactModelType.ALMGREN_CHRISS,
            "almgren_chriss": ImpactModelType.ALMGREN_CHRISS,
            "gatheral": ImpactModelType.GATHERAL,
            "composite": ImpactModelType.COMPOSITE,
            "linear": ImpactModelType.LINEAR,
        }
        model_enum = type_map.get(model_type.lower(), ImpactModelType.ALMGREN_CHRISS)
    else:
        model_enum = model_type

    # Create model
    if model_enum == ImpactModelType.KYLE_LAMBDA:
        return KyleLambdaModel(
            lambda_coef=kwargs.get("lambda_coef", 0.0001),
            permanent_fraction=kwargs.get("permanent_fraction", 0.5),
            decay_type=kwargs.get("decay_type", DecayType.EXPONENTIAL),
            decay_tau_ms=default_params.tau_ms,
        )
    elif model_enum == ImpactModelType.GATHERAL:
        return GatheralModel(params=default_params)
    elif model_enum == ImpactModelType.COMPOSITE:
        return CompositeImpactModel()
    else:  # Default: Almgren-Chriss
        return AlmgrenChrissModel(params=default_params)


def create_impact_tracker(
    model_type: str = "almgren_chriss",
    asset_class: str = "equity",
    **kwargs,
) -> ImpactTracker:
    """
    Factory function to create an impact tracker with model.

    Args:
        model_type: Model type
        asset_class: Asset class
        **kwargs: Model parameters

    Returns:
        ImpactTracker instance
    """
    model = create_impact_model(model_type, asset_class, **kwargs)
    return ImpactTracker(model=model)
