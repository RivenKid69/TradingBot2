# -*- coding: utf-8 -*-
"""
futures_env.py
Futures trading environment wrapper for TradingEnv (Phase 8).

This wrapper provides futures-specific adjustments:
- Leverage control and margin tracking
- Funding payment integration in rewards (crypto perpetuals)
- Liquidation handling and penalty
- Margin ratio observation augmentation
- ADL (Auto-Deleveraging) risk tracking

Supports:
- Crypto Perpetual (Binance USDT-M, Bybit) - funding every 8h
- Crypto Quarterly (expiring contracts)
- CME Index/Commodity/Currency Futures (via IB)

Usage:
    from wrappers.futures_env import FuturesTradingEnv, create_futures_env

    env = TradingEnv(df, asset_class="crypto_futures")
    wrapped_env = FuturesTradingEnv(
        env,
        initial_leverage=10,
        max_leverage=50,
        margin_mode="cross",
        include_funding_in_reward=True,
    )

    obs, info = wrapped_env.reset()
    obs, reward, terminated, truncated, info = wrapped_env.step(action)

References:
- Binance Futures API: https://binance-docs.github.io/apidocs/futures/en/
- CME Group Contract Specs: https://www.cmegroup.com/trading/equity-index/
- Phase 8: Training Integration from FUTURES_INTEGRATION_PLAN.md

Author: Sultan Suyunbayev
Date: 2025-12-02
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core_futures import (
    FuturesPosition,
    FuturesContractSpec,
    FuturesType,
    MarginMode,
    PositionSide,
    FundingPayment,
    MarginRequirement,
    LeverageBracket,
)
from services.futures_feature_flags import (
    FuturesFeatureFlags,
    FuturesFeature,
    get_global_flags,
    feature_flag,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Standard funding intervals (UTC hours)
BINANCE_FUNDING_TIMES = [0, 8, 16]  # 00:00, 08:00, 16:00 UTC
BYBIT_FUNDING_TIMES = [0, 8, 16]  # Same as Binance

# Default liquidation penalty (severe negative reward)
DEFAULT_LIQUIDATION_PENALTY = -10.0

# Margin ratio thresholds
MARGIN_WARNING_THRESHOLD = 1.5  # Warning at 150% margin ratio
MARGIN_DANGER_THRESHOLD = 1.2  # Danger at 120%
MARGIN_CRITICAL_THRESHOLD = 1.05  # Critical at 105%
MARGIN_LIQUIDATION_THRESHOLD = 1.0  # Liquidation at 100%

# Default leverage brackets (Binance BTCUSDT style)
DEFAULT_LEVERAGE_BRACKETS = [
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
    LeverageBracket(
        bracket=3,
        notional_cap=Decimal("1000000"),
        maint_margin_rate=Decimal("0.01"),
        max_leverage=50,
        cum_maintenance=Decimal("1300"),
    ),
    LeverageBracket(
        bracket=4,
        notional_cap=Decimal("5000000"),
        maint_margin_rate=Decimal("0.025"),
        max_leverage=20,
        cum_maintenance=Decimal("16300"),
    ),
    LeverageBracket(
        bracket=5,
        notional_cap=Decimal("20000000"),
        maint_margin_rate=Decimal("0.05"),
        max_leverage=10,
        cum_maintenance=Decimal("141300"),
    ),
]


# =============================================================================
# FUNDING RATE PROVIDER
# =============================================================================


@dataclass
class FundingRateData:
    """Funding rate data point."""

    timestamp_ms: int
    funding_rate: Decimal
    mark_price: Optional[Decimal] = None
    symbol: str = ""


class FundingRateProvider:
    """
    Provider for funding rate data.

    Supports:
    - Historical funding rate lookup from preloaded data
    - Time-series funding rate for backtesting
    - Default constant rate for simple simulations
    """

    def __init__(
        self,
        rates: Optional[Dict[int, FundingRateData]] = None,
        default_rate: Decimal = Decimal("0.0001"),  # 0.01% default
    ):
        """
        Initialize funding rate provider.

        Args:
            rates: Pre-loaded funding rates by timestamp
            default_rate: Default rate if timestamp not found
        """
        self._rates: Dict[int, FundingRateData] = rates or {}
        self._rate_list: List[FundingRateData] = []
        self._default_rate = default_rate

        if rates:
            self._rate_list = sorted(rates.values(), key=lambda x: x.timestamp_ms)

    @classmethod
    def from_dataframe(
        cls,
        df,  # pandas DataFrame
        rate_col: str = "funding_rate",
        timestamp_col: str = "timestamp_ms",
        symbol: str = "",
    ) -> "FundingRateProvider":
        """
        Create provider from pandas DataFrame.

        Args:
            df: DataFrame with funding rate data
            rate_col: Column name for funding rate
            timestamp_col: Column name for timestamp
            symbol: Symbol for the rates

        Returns:
            FundingRateProvider instance
        """
        rates = {}
        for _, row in df.iterrows():
            ts = int(row[timestamp_col])
            rates[ts] = FundingRateData(
                timestamp_ms=ts,
                funding_rate=Decimal(str(row[rate_col])),
                mark_price=Decimal(str(row.get("mark_price", 0))) if "mark_price" in row else None,
                symbol=symbol,
            )
        return cls(rates=rates)

    def get_funding_rate(self, timestamp_ms: int) -> FundingRateData:
        """
        Get funding rate at or before the given timestamp.

        Args:
            timestamp_ms: Timestamp in milliseconds

        Returns:
            FundingRateData (uses most recent rate before timestamp)
        """
        if not self._rate_list:
            return FundingRateData(
                timestamp_ms=timestamp_ms,
                funding_rate=self._default_rate,
            )

        # Binary search for most recent rate
        left, right = 0, len(self._rate_list) - 1
        result = self._rate_list[0]

        while left <= right:
            mid = (left + right) // 2
            if self._rate_list[mid].timestamp_ms <= timestamp_ms:
                result = self._rate_list[mid]
                left = mid + 1
            else:
                right = mid - 1

        return result

    def add_rate(self, rate: FundingRateData) -> None:
        """Add a funding rate data point."""
        self._rates[rate.timestamp_ms] = rate
        self._rate_list = sorted(self._rates.values(), key=lambda x: x.timestamp_ms)


# =============================================================================
# MARGIN CALCULATOR
# =============================================================================


class MarginCalculator:
    """
    Calculator for futures margin requirements.

    Supports both:
    - Tiered brackets (Binance style)
    - Flat percentage (CME style approximation)
    """

    def __init__(
        self,
        brackets: Optional[List[LeverageBracket]] = None,
        flat_initial_pct: Optional[Decimal] = None,
        flat_maint_pct: Optional[Decimal] = None,
    ):
        """
        Initialize margin calculator.

        Args:
            brackets: Leverage brackets for tiered margin (Binance)
            flat_initial_pct: Flat initial margin percentage (CME)
            flat_maint_pct: Flat maintenance margin percentage (CME)
        """
        self._brackets = brackets or DEFAULT_LEVERAGE_BRACKETS
        self._flat_initial = flat_initial_pct
        self._flat_maint = flat_maint_pct

    def calculate_margin(
        self,
        notional: Decimal,
        leverage: int,
        use_brackets: bool = True,
    ) -> MarginRequirement:
        """
        Calculate margin requirements for a position.

        Args:
            notional: Position notional value
            leverage: Current leverage setting
            use_brackets: Use tiered brackets (Binance) or flat (CME)

        Returns:
            MarginRequirement with initial and maintenance margin
        """
        if not use_brackets or self._flat_initial is not None:
            # Flat margin calculation
            initial_pct = self._flat_initial or Decimal("5.0")
            maint_pct = self._flat_maint or Decimal("4.0")
            initial = notional * initial_pct / 100
            maintenance = notional * maint_pct / 100
        else:
            # Tiered bracket calculation
            bracket = self._get_bracket(notional)
            initial = notional / Decimal(leverage)
            maintenance = notional * bracket.maint_margin_rate + bracket.cum_maintenance

        return MarginRequirement(
            initial=initial,
            maintenance=maintenance,
        )

    def get_max_leverage(self, notional: Decimal) -> int:
        """Get maximum allowed leverage for a given notional."""
        bracket = self._get_bracket(notional)
        return bracket.max_leverage

    def _get_bracket(self, notional: Decimal) -> LeverageBracket:
        """Get the appropriate bracket for notional size."""
        for bracket in self._brackets:
            if bracket.notional_cap is None or notional <= bracket.notional_cap:
                return bracket
        return self._brackets[-1]


# =============================================================================
# LIQUIDATION CALCULATOR
# =============================================================================


class LiquidationCalculator:
    """
    Calculator for liquidation price estimation.

    Uses the formula:
    - Long: liq_price = entry_price * (1 - 1/leverage + maint_margin_rate)
    - Short: liq_price = entry_price * (1 + 1/leverage - maint_margin_rate)
    """

    def __init__(
        self,
        margin_calculator: Optional[MarginCalculator] = None,
    ):
        """
        Initialize liquidation calculator.

        Args:
            margin_calculator: MarginCalculator for margin rates
        """
        self._margin_calc = margin_calculator or MarginCalculator()

    def calculate_liquidation_price(
        self,
        entry_price: Decimal,
        leverage: int,
        is_long: bool,
        maint_margin_rate: Decimal = Decimal("0.004"),
        additional_margin: Decimal = Decimal("0"),
    ) -> Decimal:
        """
        Calculate estimated liquidation price.

        Args:
            entry_price: Position entry price
            leverage: Current leverage
            is_long: True for long, False for short
            maint_margin_rate: Maintenance margin rate
            additional_margin: Extra margin added to position

        Returns:
            Estimated liquidation price
        """
        if leverage <= 0 or entry_price <= 0:
            return Decimal("0")

        # Initial margin rate = 1/leverage
        imr = Decimal("1") / Decimal(leverage)

        # Additional margin effect (reduces liquidation risk)
        if additional_margin > 0 and entry_price > 0:
            additional_margin_rate = additional_margin / entry_price
        else:
            additional_margin_rate = Decimal("0")

        if is_long:
            # Long position liquidates when price drops
            liq_price = entry_price * (1 - imr + maint_margin_rate - additional_margin_rate)
        else:
            # Short position liquidates when price rises
            liq_price = entry_price * (1 + imr - maint_margin_rate + additional_margin_rate)

        return max(Decimal("0"), liq_price)

    def is_liquidated(
        self,
        current_price: Decimal,
        liquidation_price: Decimal,
        is_long: bool,
    ) -> bool:
        """
        Check if position is liquidated at current price.

        Args:
            current_price: Current mark/market price
            liquidation_price: Calculated liquidation price
            is_long: True for long, False for short

        Returns:
            True if position would be liquidated
        """
        if liquidation_price <= 0:
            return False

        if is_long:
            return current_price <= liquidation_price
        else:
            return current_price >= liquidation_price


# =============================================================================
# FUTURES TRADING ENVIRONMENT WRAPPER
# =============================================================================


@dataclass
class FuturesEnvConfig:
    """Configuration for FuturesTradingEnv."""

    initial_leverage: int = 10
    max_leverage: int = 50
    margin_mode: str = "cross"  # "cross" or "isolated"
    include_funding_in_reward: bool = True
    liquidation_penalty: float = DEFAULT_LIQUIDATION_PENALTY
    funding_times_utc: List[int] = field(default_factory=lambda: BINANCE_FUNDING_TIMES)
    augment_observation: bool = True
    track_adl_risk: bool = True
    use_mark_price: bool = True
    futures_type: str = "crypto_perp"  # crypto_perp, crypto_quarterly, index, commodity
    symbol: str = ""


class FuturesTradingEnv(gym.Wrapper):
    """
    Futures trading environment with leverage, margin, and funding.

    Extends TradingEnv with futures-specific mechanics:
    1. Leverage control (tiered for crypto, flat for CME)
    2. Margin tracking (initial + maintenance)
    3. Funding payment integration (perpetuals)
    4. Liquidation handling
    5. ADL (Auto-Deleveraging) risk tracking

    Action Space:
    - Original: position size [-1, 1]
    - Extended (optional): [position_size, leverage_target]

    Observation Space:
    - Original features
    - + margin_ratio (augmented)
    - + funding_rate (augmented)
    - + liquidation_distance (augmented)

    Args:
        env: The underlying TradingEnv
        initial_leverage: Starting leverage (default: 10)
        max_leverage: Maximum allowed leverage (default: 50)
        margin_mode: "cross" or "isolated" margin
        include_funding_in_reward: Include funding P&L in reward
        liquidation_penalty: Reward penalty on liquidation
        funding_provider: Optional FundingRateProvider
        margin_calculator: Optional MarginCalculator
        config: Optional FuturesEnvConfig for full configuration
    """

    def __init__(
        self,
        env: gym.Env,
        initial_leverage: int = 10,
        max_leverage: int = 50,
        margin_mode: str = "cross",
        include_funding_in_reward: bool = True,
        liquidation_penalty: float = DEFAULT_LIQUIDATION_PENALTY,
        funding_provider: Optional[FundingRateProvider] = None,
        margin_calculator: Optional[MarginCalculator] = None,
        liquidation_calculator: Optional[LiquidationCalculator] = None,
        config: Optional[FuturesEnvConfig] = None,
        feature_flags: Optional[FuturesFeatureFlags] = None,
    ):
        super().__init__(env)

        # Use config if provided, otherwise use individual parameters
        if config is not None:
            self._leverage = config.initial_leverage
            self._max_leverage = config.max_leverage
            self._margin_mode = MarginMode(config.margin_mode.upper())
            self._include_funding = config.include_funding_in_reward
            self._liquidation_penalty = config.liquidation_penalty
            self._funding_times = config.funding_times_utc
            self._augment_obs = config.augment_observation
            self._track_adl = config.track_adl_risk
            self._use_mark_price = config.use_mark_price
            self._futures_type = config.futures_type
            self._symbol = config.symbol
        else:
            self._leverage = initial_leverage
            self._max_leverage = max_leverage
            self._margin_mode = MarginMode(margin_mode.upper())
            self._include_funding = include_funding_in_reward
            self._liquidation_penalty = liquidation_penalty
            self._funding_times = BINANCE_FUNDING_TIMES
            self._augment_obs = True
            self._track_adl = True
            self._use_mark_price = True
            self._futures_type = "crypto_perp"
            self._symbol = ""

        # Calculators
        self._funding_provider = funding_provider or FundingRateProvider()
        self._margin_calculator = margin_calculator or MarginCalculator()
        self._liquidation_calculator = liquidation_calculator or LiquidationCalculator(
            margin_calculator=self._margin_calculator
        )

        # Feature flags
        self._feature_flags = feature_flags or get_global_flags()

        # Position state (mutable during episode)
        self._position: Optional[FuturesPosition] = None
        self._entry_price: Decimal = Decimal("0")
        self._position_qty: Decimal = Decimal("0")
        self._liquidation_price: Decimal = Decimal("0")

        # Funding tracking
        self._realized_funding: Decimal = Decimal("0")
        self._funding_count: int = 0
        self._last_funding_ts: int = 0

        # Margin tracking
        self._initial_margin: Decimal = Decimal("0")
        self._maint_margin: Decimal = Decimal("0")
        self._margin_ratio: float = 1.0

        # Episode state
        self._last_timestamp: int = 0
        self._was_liquidated: bool = False
        self._total_pnl: Decimal = Decimal("0")
        self._unrealized_pnl: Decimal = Decimal("0")

        # Observation augmentation
        self._n_augmented_features = 4 if self._augment_obs else 0
        if self._augment_obs:
            self._modify_observation_space()

        logger.debug(
            f"FuturesTradingEnv initialized: leverage={self._leverage}/{self._max_leverage}, "
            f"margin_mode={self._margin_mode.value}, funding={self._include_funding}, "
            f"futures_type={self._futures_type}"
        )

    def _modify_observation_space(self) -> None:
        """Modify observation space to include futures features."""
        if not isinstance(self.observation_space, spaces.Box):
            logger.warning("Cannot augment non-Box observation space")
            return

        old_shape = self.observation_space.shape
        new_shape = (old_shape[0] + self._n_augmented_features,)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=new_shape,
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and initialize futures state."""
        obs, info = self.env.reset(seed=seed, options=options)

        # Reset futures-specific state
        self._position = None
        self._entry_price = Decimal("0")
        self._position_qty = Decimal("0")
        self._liquidation_price = Decimal("0")

        self._realized_funding = Decimal("0")
        self._funding_count = 0
        self._last_funding_ts = 0

        self._initial_margin = Decimal("0")
        self._maint_margin = Decimal("0")
        self._margin_ratio = 1.0

        self._last_timestamp = info.get("timestamp_ms", info.get("timestamp", 0))
        self._was_liquidated = False
        self._total_pnl = Decimal("0")
        self._unrealized_pnl = Decimal("0")

        # Get symbol from info or env
        if not self._symbol:
            self._symbol = info.get("symbol", getattr(self.env, "symbol", "BTCUSDT"))

        # Augment observation
        if self._augment_obs:
            obs = self._augment_observation(obs, info)

        # Enrich info
        info = self._enrich_info(info)

        return obs, info

    def step(
        self,
        action: Union[np.ndarray, float, int],
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute step with futures-specific mechanics."""
        # Parse action (position_size, optional leverage)
        if isinstance(action, np.ndarray) and len(action) > 1:
            position_target = float(action[0])
            leverage_target = int(np.clip(action[1], 1, self._max_leverage))
            self._leverage = leverage_target
        else:
            position_target = float(action[0]) if isinstance(action, np.ndarray) else float(action)

        # Execute underlying environment step
        obs, reward, terminated, truncated, info = self.env.step(position_target)

        # Get current state
        current_timestamp = info.get("timestamp_ms", info.get("timestamp", self._last_timestamp))
        current_position = info.get("signal_pos_next", 0.0)
        mark_price = Decimal(str(info.get("mark_price", info.get("close", 0))))

        # Update position tracking
        self._update_position_state(current_position, mark_price, info)

        # Check for liquidation (before funding to avoid paying funding on liquidated position)
        if self._check_liquidation(mark_price):
            terminated = True
            reward = self._liquidation_penalty
            self._was_liquidated = True
            info["liquidated"] = True
            info["liquidation_price"] = float(self._liquidation_price)
            logger.info(
                f"Position liquidated at {mark_price}, "
                f"liquidation_price={self._liquidation_price}"
            )

        # Apply funding if due (perpetuals only)
        funding_payment = Decimal("0")
        if (
            self._include_funding
            and self._futures_type in ("crypto_perp", "perpetual")
            and not self._was_liquidated
        ):
            funding_payment = self._apply_funding(current_timestamp, mark_price)

        # Adjust reward with funding
        adjusted_reward = reward
        if funding_payment != 0:
            adjusted_reward += float(funding_payment)

        # Calculate margin and PnL
        self._update_margin_state(mark_price)
        self._calculate_pnl(mark_price)

        # Augment observation
        if self._augment_obs:
            obs = self._augment_observation(obs, info)

        # Update timestamps
        self._last_timestamp = current_timestamp

        # Enrich info with futures data
        info = self._enrich_info(info)
        info["funding_payment"] = float(funding_payment)
        info["cumulative_funding"] = float(self._realized_funding)
        info["funding_count"] = self._funding_count
        info["margin_ratio"] = self._margin_ratio
        info["liquidation_distance"] = self._calculate_liquidation_distance(mark_price)
        info["unrealized_pnl"] = float(self._unrealized_pnl)
        info["leverage"] = self._leverage
        info["reward_adjustment"] = adjusted_reward - reward

        return obs, adjusted_reward, terminated, truncated, info

    def _update_position_state(
        self,
        position_signal: float,
        mark_price: Decimal,
        info: Dict[str, Any],
    ) -> None:
        """Update internal position state based on signal."""
        old_qty = self._position_qty
        new_qty = Decimal(str(position_signal))

        # Track entry price updates
        if abs(new_qty) > abs(old_qty):
            # Increasing position - update entry price (simplified average)
            if old_qty == 0:
                self._entry_price = mark_price
            else:
                # Weighted average entry
                old_notional = abs(old_qty) * self._entry_price
                new_notional = abs(new_qty - old_qty) * mark_price
                self._entry_price = (old_notional + new_notional) / abs(new_qty)
        elif abs(new_qty) < abs(old_qty) and new_qty != 0:
            # Reducing position - entry price stays same
            pass
        elif new_qty == 0:
            # Position closed
            self._entry_price = Decimal("0")

        self._position_qty = new_qty

        # Update liquidation price
        if abs(new_qty) > 0:
            self._liquidation_price = self._liquidation_calculator.calculate_liquidation_price(
                entry_price=self._entry_price,
                leverage=self._leverage,
                is_long=(new_qty > 0),
                maint_margin_rate=Decimal("0.004"),  # Default, should come from bracket
            )
        else:
            self._liquidation_price = Decimal("0")

    def _update_margin_state(self, mark_price: Decimal) -> None:
        """Update margin requirements and ratio."""
        if self._position_qty == 0 or mark_price == 0:
            self._initial_margin = Decimal("0")
            self._maint_margin = Decimal("0")
            self._margin_ratio = float("inf")  # No position = infinite margin ratio
            return

        notional = abs(self._position_qty) * mark_price
        margin_req = self._margin_calculator.calculate_margin(
            notional=notional,
            leverage=self._leverage,
        )

        self._initial_margin = margin_req.initial
        self._maint_margin = margin_req.maintenance

        # Margin ratio = equity / maintenance margin
        # For simplicity, assume equity = initial margin + unrealized PnL
        equity = self._initial_margin + self._unrealized_pnl
        if self._maint_margin > 0:
            self._margin_ratio = float(equity / self._maint_margin)
        else:
            self._margin_ratio = float("inf")

    def _calculate_pnl(self, mark_price: Decimal) -> None:
        """Calculate unrealized P&L."""
        if self._position_qty == 0 or self._entry_price == 0:
            self._unrealized_pnl = Decimal("0")
            return

        price_diff = mark_price - self._entry_price
        self._unrealized_pnl = price_diff * self._position_qty

    def _apply_funding(
        self,
        current_timestamp: int,
        mark_price: Decimal,
    ) -> Decimal:
        """
        Apply funding payment if due.

        Args:
            current_timestamp: Current timestamp in ms
            mark_price: Current mark price

        Returns:
            Funding payment amount (positive = received, negative = paid)
        """
        if self._position_qty == 0:
            return Decimal("0")

        # Check if we crossed a funding time
        funding_crossed = self._check_funding_time(
            self._last_funding_ts or self._last_timestamp,
            current_timestamp,
        )

        if not funding_crossed:
            return Decimal("0")

        # Get funding rate
        rate_data = self._funding_provider.get_funding_rate(current_timestamp)
        funding_rate = rate_data.funding_rate

        # Calculate payment
        # Positive rate + long position = pay funding
        # Positive rate + short position = receive funding
        position_notional = abs(self._position_qty) * mark_price

        if self._position_qty > 0:  # Long
            # Long pays if rate positive, receives if rate negative
            payment = -position_notional * funding_rate
        else:  # Short
            # Short receives if rate positive, pays if rate negative
            payment = position_notional * funding_rate

        self._realized_funding += payment
        self._funding_count += 1
        self._last_funding_ts = current_timestamp

        logger.debug(
            f"Funding applied: rate={float(funding_rate):.6f}, "
            f"payment={float(payment):.4f}, position={float(self._position_qty):.4f}"
        )

        return payment

    def _check_funding_time(
        self,
        prev_ts: int,
        curr_ts: int,
    ) -> bool:
        """Check if a funding time was crossed between timestamps."""
        if prev_ts == 0 or curr_ts <= prev_ts:
            return False

        # Convert to datetime
        prev_dt = datetime.fromtimestamp(prev_ts / 1000, tz=timezone.utc)
        curr_dt = datetime.fromtimestamp(curr_ts / 1000, tz=timezone.utc)

        # Check each funding time
        for funding_hour in self._funding_times:
            # Check if we crossed this funding hour
            prev_funding = prev_dt.replace(hour=funding_hour, minute=0, second=0, microsecond=0)
            curr_funding = curr_dt.replace(hour=funding_hour, minute=0, second=0, microsecond=0)

            # If prev_dt was before funding time and curr_dt is at/after
            if prev_dt < prev_funding <= curr_dt:
                return True

            # Also check next day's funding time if we crossed midnight
            if prev_dt.date() != curr_dt.date():
                next_day_funding = curr_funding
                if prev_dt < next_day_funding <= curr_dt:
                    return True

        return False

    def _check_liquidation(self, mark_price: Decimal) -> bool:
        """Check if position would be liquidated."""
        if self._position_qty == 0 or self._liquidation_price == 0:
            return False

        return self._liquidation_calculator.is_liquidated(
            current_price=mark_price,
            liquidation_price=self._liquidation_price,
            is_long=(self._position_qty > 0),
        )

    def _calculate_liquidation_distance(self, mark_price: Decimal) -> float:
        """Calculate distance to liquidation as percentage."""
        if self._position_qty == 0 or self._liquidation_price == 0 or mark_price == 0:
            return float("inf")

        distance = abs(mark_price - self._liquidation_price) / mark_price * 100
        return float(distance)

    def _augment_observation(
        self,
        obs: np.ndarray,
        info: Dict[str, Any],
    ) -> np.ndarray:
        """Add futures-specific features to observation."""
        if not self._augment_obs:
            return obs

        # Get current mark price
        mark_price = Decimal(str(info.get("mark_price", info.get("close", 1))))

        # Augmented features (normalized)
        augmented = np.array(
            [
                # Margin ratio (clipped and normalized to [-1, 1])
                np.clip((self._margin_ratio - 1.0) / 2.0, -1.0, 1.0),
                # Funding rate (from info or default)
                np.clip(float(info.get("funding_rate", 0)) * 1000, -1.0, 1.0),
                # Liquidation distance (normalized log)
                np.clip(
                    math.log1p(self._calculate_liquidation_distance(mark_price)) / 5.0, 0.0, 1.0
                ),
                # Current leverage (normalized)
                np.clip(self._leverage / self._max_leverage, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

        return np.concatenate([obs, augmented])

    def _enrich_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Add futures-specific information to info dict."""
        info["futures_type"] = self._futures_type
        info["margin_mode"] = self._margin_mode.value
        info["max_leverage"] = self._max_leverage
        info["current_leverage"] = self._leverage
        info["initial_margin"] = float(self._initial_margin)
        info["maint_margin"] = float(self._maint_margin)
        info["liquidation_price"] = float(self._liquidation_price)
        info["entry_price"] = float(self._entry_price)
        info["position_qty"] = float(self._position_qty)
        info["was_liquidated"] = self._was_liquidated
        info["symbol"] = self._symbol

        # Margin status
        if self._margin_ratio >= MARGIN_WARNING_THRESHOLD:
            info["margin_status"] = "healthy"
        elif self._margin_ratio >= MARGIN_DANGER_THRESHOLD:
            info["margin_status"] = "warning"
        elif self._margin_ratio >= MARGIN_CRITICAL_THRESHOLD:
            info["margin_status"] = "danger"
        else:
            info["margin_status"] = "critical"

        return info

    @property
    def leverage(self) -> int:
        """Current leverage setting."""
        return self._leverage

    @leverage.setter
    def leverage(self, value: int) -> None:
        """Set leverage (clamped to max)."""
        self._leverage = max(1, min(value, self._max_leverage))

    @property
    def margin_mode(self) -> MarginMode:
        """Current margin mode."""
        return self._margin_mode

    @property
    def funding_rate_provider(self) -> FundingRateProvider:
        """Get funding rate provider."""
        return self._funding_provider


# =============================================================================
# LEVERAGE WRAPPER (Simple leverage enforcement)
# =============================================================================


class FuturesLeverageWrapper(gym.Wrapper):
    """
    Simple wrapper that enforces leverage limits.

    For use when full futures mechanics are not needed,
    just leverage constraint enforcement.
    """

    def __init__(
        self,
        env: gym.Env,
        max_leverage: int = 50,
        initial_leverage: int = 10,
    ):
        super().__init__(env)
        self._max_leverage = max_leverage
        self._leverage = initial_leverage

    def step(self, action: Union[np.ndarray, float]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["leverage"] = self._leverage
        info["max_leverage"] = self._max_leverage
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(**kwargs)
        info["leverage"] = self._leverage
        info["max_leverage"] = self._max_leverage
        return obs, info


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_futures_env(
    base_env: gym.Env,
    config: Optional[Dict[str, Any]] = None,
    funding_provider: Optional[FundingRateProvider] = None,
    feature_flags: Optional[FuturesFeatureFlags] = None,
) -> FuturesTradingEnv:
    """
    Factory function to create a FuturesTradingEnv.

    Args:
        base_env: The underlying TradingEnv
        config: Configuration dictionary
        funding_provider: Optional FundingRateProvider
        feature_flags: Optional FuturesFeatureFlags

    Returns:
        Configured FuturesTradingEnv
    """
    config = config or {}

    env_config = FuturesEnvConfig(
        initial_leverage=config.get("initial_leverage", 10),
        max_leverage=config.get("max_leverage", 50),
        margin_mode=config.get("margin_mode", "cross"),
        include_funding_in_reward=config.get("include_funding_in_reward", True),
        liquidation_penalty=config.get("liquidation_penalty", DEFAULT_LIQUIDATION_PENALTY),
        funding_times_utc=config.get("funding_times_utc", BINANCE_FUNDING_TIMES),
        augment_observation=config.get("augment_observation", True),
        track_adl_risk=config.get("track_adl_risk", True),
        use_mark_price=config.get("use_mark_price", True),
        futures_type=config.get("futures_type", "crypto_perp"),
        symbol=config.get("symbol", ""),
    )

    return FuturesTradingEnv(
        env=base_env,
        config=env_config,
        funding_provider=funding_provider,
        feature_flags=feature_flags,
    )


def create_cme_futures_env(
    base_env: gym.Env,
    config: Optional[Dict[str, Any]] = None,
    feature_flags: Optional[FuturesFeatureFlags] = None,
) -> FuturesTradingEnv:
    """
    Factory function to create a FuturesTradingEnv for CME futures.

    CME futures don't have funding, use flat margin, and have
    daily settlement instead.

    Args:
        base_env: The underlying TradingEnv
        config: Configuration dictionary
        feature_flags: Optional FuturesFeatureFlags

    Returns:
        Configured FuturesTradingEnv for CME futures
    """
    config = config or {}

    env_config = FuturesEnvConfig(
        initial_leverage=config.get("initial_leverage", 10),
        max_leverage=config.get("max_leverage", 20),  # CME typically lower
        margin_mode="span" if config.get("use_span", False) else "cross",
        include_funding_in_reward=False,  # No funding for CME
        liquidation_penalty=config.get("liquidation_penalty", DEFAULT_LIQUIDATION_PENALTY),
        funding_times_utc=[],  # No funding times
        augment_observation=config.get("augment_observation", True),
        track_adl_risk=False,  # No ADL for CME
        use_mark_price=False,  # CME uses last price
        futures_type=config.get("futures_type", "index"),
        symbol=config.get("symbol", "ES"),
    )

    # Use flat margin calculator for CME
    margin_calc = MarginCalculator(
        flat_initial_pct=Decimal(str(config.get("initial_margin_pct", "5.0"))),
        flat_maint_pct=Decimal(str(config.get("maint_margin_pct", "4.0"))),
    )

    return FuturesTradingEnv(
        env=base_env,
        config=env_config,
        margin_calculator=margin_calc,
        feature_flags=feature_flags,
    )
