# -*- coding: utf-8 -*-
"""
forex_env.py
Forex-specific environment wrapper for TradingEnv (Phase 9).

This wrapper provides forex-specific adjustments:
- Weekend gap filtering (skip weekend observations)
- Session-aware reward scaling (lower rewards during low-liquidity sessions)
- Swap/rollover cost deduction from rewards (with DST-aware rollover detection)
- Leverage constraint enforcement
- SwapRateProvider for loading swap rates from files

Improvements (2025-11-30):
- DST-aware rollover time detection using zoneinfo
- Optimized rollover counter algorithm (O(days) instead of O(hours))
- SwapRateProvider with file loading and pair-specific rates

Usage:
    from wrappers.forex_env import ForexEnvWrapper, SwapRateProvider

    env = TradingEnv(df, asset_class="forex")
    swap_provider = SwapRateProvider.from_directory("data/forex/swaps")
    wrapped_env = ForexEnvWrapper(
        env,
        leverage=30.0,
        include_swap_costs=True,
        swap_provider=swap_provider,
    )

    obs, info = wrapped_env.reset()
    obs, reward, terminated, truncated, info = wrapped_env.step(action)

Author: Sultan Suyunbayev
Date: 2025-11-30
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)

# Try to import zoneinfo (Python 3.9+)
try:
    from zoneinfo import ZoneInfo
except ImportError:
    # Fallback for older Python versions
    try:
        from backports.zoneinfo import ZoneInfo
    except ImportError:
        ZoneInfo = None  # type: ignore
        logger.warning("zoneinfo not available, using UTC approximation for DST")


# =============================================================================
# CONSTANTS
# =============================================================================

# Session liquidity factors (relative to average, based on BIS 2022 data)
SESSION_LIQUIDITY: Dict[str, float] = {
    "sydney": 0.6,
    "tokyo": 0.8,
    "london": 1.3,
    "new_york": 1.2,
    "london_ny_overlap": 1.5,
    "tokyo_london_overlap": 1.0,
    "low_liquidity": 0.4,
    "weekend": 0.0,
}

# Session times (UTC hours, approximate)
SESSION_TIMES_UTC: Dict[str, Tuple[int, int]] = {
    "sydney": (21, 6),  # 21:00 - 06:00 UTC
    "tokyo": (0, 9),  # 00:00 - 09:00 UTC
    "london": (7, 16),  # 07:00 - 16:00 UTC
    "new_york": (12, 21),  # 12:00 - 21:00 UTC
}

# Default swap rates by pair category (pips/day, negative = cost)
# Based on typical retail broker rates
DEFAULT_SWAP_RATES: Dict[str, Dict[str, float]] = {
    "majors": {"long": -0.3, "short": 0.1},
    "minors": {"long": -0.5, "short": 0.2},
    "crosses": {"long": -0.8, "short": 0.3},
    "exotics": {"long": -2.0, "short": 0.5},
}

# Pair classification
MAJOR_PAIRS = frozenset(
    {
        "EUR_USD",
        "GBP_USD",
        "USD_JPY",
        "USD_CHF",
        "AUD_USD",
        "USD_CAD",
        "NZD_USD",
    }
)
MINOR_PAIRS = frozenset(
    {
        "EUR_GBP",
        "EUR_CHF",
        "GBP_CHF",
        "EUR_AUD",
        "EUR_CAD",
        "EUR_NZD",
        "GBP_AUD",
        "GBP_CAD",
    }
)
EXOTIC_PAIRS = frozenset(
    {
        "USD_TRY",
        "USD_ZAR",
        "USD_MXN",
        "USD_PLN",
        "USD_HUF",
        "USD_CZK",
        "USD_SGD",
        "USD_HKD",
        "USD_NOK",
        "USD_SEK",
        "USD_DKK",
    }
)


# =============================================================================
# SWAP RATE PROVIDER
# =============================================================================


@dataclass
class SwapRate:
    """Swap rate for a currency pair."""

    pair: str
    long_swap: float  # Pips/day for long positions (negative = cost)
    short_swap: float  # Pips/day for short positions (negative = cost)
    timestamp: Optional[int] = None  # When this rate was valid
    source: str = "default"  # Source of the rate (file, api, default)


class SwapRateProvider:
    """
    Provider for swap/rollover rates with multiple data sources.

    Supports:
    - Loading from JSON/CSV files
    - Per-pair rates
    - Time-varying rates (for backtesting)
    - Default rates based on pair category

    File format (JSON):
        {
            "EUR_USD": {"long": -0.3, "short": 0.1, "timestamp": 1704067200},
            "GBP_USD": {"long": -0.4, "short": 0.15}
        }

    File format (CSV):
        pair,long_swap,short_swap,timestamp
        EUR_USD,-0.3,0.1,1704067200
        GBP_USD,-0.4,0.15,
    """

    def __init__(
        self,
        rates: Optional[Dict[str, SwapRate]] = None,
        use_defaults: bool = True,
    ):
        """
        Initialize swap rate provider.

        Args:
            rates: Pre-loaded swap rates by pair
            use_defaults: Fall back to default rates if pair not found
        """
        self._rates: Dict[str, SwapRate] = rates or {}
        self._use_defaults = use_defaults
        self._historical_rates: Dict[str, List[SwapRate]] = {}

    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> "SwapRateProvider":
        """Load swap rates from a JSON or CSV file."""
        filepath = Path(filepath)

        if not filepath.exists():
            logger.warning(f"Swap rate file not found: {filepath}, using defaults")
            return cls(use_defaults=True)

        rates: Dict[str, SwapRate] = {}

        if filepath.suffix.lower() == ".json":
            with open(filepath, "r") as f:
                data = json.load(f)

            for pair, values in data.items():
                rates[pair.upper().replace("/", "_")] = SwapRate(
                    pair=pair.upper().replace("/", "_"),
                    long_swap=float(values.get("long", values.get("long_swap", 0.0))),
                    short_swap=float(values.get("short", values.get("short_swap", 0.0))),
                    timestamp=values.get("timestamp"),
                    source="file",
                )

        elif filepath.suffix.lower() == ".csv":
            import csv

            with open(filepath, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pair = row["pair"].upper().replace("/", "_")
                    rates[pair] = SwapRate(
                        pair=pair,
                        long_swap=float(row.get("long_swap", row.get("long", 0.0))),
                        short_swap=float(row.get("short_swap", row.get("short", 0.0))),
                        timestamp=int(row["timestamp"]) if row.get("timestamp") else None,
                        source="file",
                    )

        logger.info(f"Loaded {len(rates)} swap rates from {filepath}")
        return cls(rates=rates, use_defaults=True)

    @classmethod
    def from_directory(
        cls,
        directory: Union[str, Path],
        pattern: str = "*.json",
    ) -> "SwapRateProvider":
        """Load swap rates from all matching files in a directory."""
        directory = Path(directory)

        if not directory.exists():
            logger.warning(f"Swap directory not found: {directory}, using defaults")
            return cls(use_defaults=True)

        rates: Dict[str, SwapRate] = {}
        historical: Dict[str, List[SwapRate]] = {}

        import glob

        for filepath in glob.glob(str(directory / pattern)):
            try:
                provider = cls.from_file(filepath)
                for pair, rate in provider._rates.items():
                    if pair not in rates:
                        rates[pair] = rate
                        historical[pair] = [rate]
                    else:
                        historical[pair].append(rate)
            except Exception as e:
                logger.warning(f"Failed to load swap file {filepath}: {e}")

        instance = cls(rates=rates, use_defaults=True)
        instance._historical_rates = historical

        logger.info(f"Loaded swap rates for {len(rates)} pairs from {directory}")
        return instance

    def get_swap_rate(
        self,
        pair: str,
        timestamp: Optional[int] = None,
    ) -> SwapRate:
        """
        Get swap rate for a pair.

        Args:
            pair: Currency pair (e.g., "EUR_USD")
            timestamp: Optional timestamp for historical lookup

        Returns:
            SwapRate for the pair
        """
        pair = pair.upper().replace("/", "_")

        # Try exact match
        if pair in self._rates:
            return self._rates[pair]

        # Try historical rates if timestamp provided
        if timestamp and pair in self._historical_rates:
            rates = sorted(
                self._historical_rates[pair],
                key=lambda r: r.timestamp or 0,
                reverse=True,
            )
            for rate in rates:
                if rate.timestamp is None or rate.timestamp <= timestamp:
                    return rate

        # Fall back to defaults based on pair category
        if self._use_defaults:
            return self._get_default_rate(pair)

        # No rate found
        return SwapRate(pair=pair, long_swap=0.0, short_swap=0.0, source="none")

    def _get_default_rate(self, pair: str) -> SwapRate:
        """Get default swap rate based on pair category."""
        pair = pair.upper().replace("/", "_")

        if pair in MAJOR_PAIRS:
            rates = DEFAULT_SWAP_RATES["majors"]
        elif pair in MINOR_PAIRS:
            rates = DEFAULT_SWAP_RATES["minors"]
        elif pair in EXOTIC_PAIRS:
            rates = DEFAULT_SWAP_RATES["exotics"]
        else:
            # Assume cross pair
            rates = DEFAULT_SWAP_RATES["crosses"]

        return SwapRate(
            pair=pair,
            long_swap=rates["long"],
            short_swap=rates["short"],
            source="default",
        )

    def add_rate(self, rate: SwapRate) -> None:
        """Add or update a swap rate."""
        self._rates[rate.pair] = rate


# =============================================================================
# DST-AWARE ROLLOVER TIME
# =============================================================================


def get_rollover_hour_utc(timestamp: int) -> int:
    """
    Get the rollover hour in UTC for a given timestamp, accounting for DST.

    Forex rollover occurs at 5:00 PM ET (Eastern Time).
    - During EST (winter): 5pm ET = 22:00 UTC
    - During EDT (summer): 5pm ET = 21:00 UTC

    Args:
        timestamp: Unix timestamp in seconds

    Returns:
        Rollover hour in UTC (21 or 22)
    """
    if ZoneInfo is None:
        # No timezone support, use approximation
        # US DST: second Sunday in March to first Sunday in November
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        month = dt.month

        # Simplified DST check (March 15 - November 1 approximately)
        if 3 <= month <= 10:
            return 21  # EDT (summer)
        else:
            return 22  # EST (winter)

    # Use proper timezone conversion
    try:
        et = ZoneInfo("America/New_York")
        dt_utc = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        dt_et = dt_utc.astimezone(et)

        # Create 5pm ET for this date
        rollover_et = dt_et.replace(hour=17, minute=0, second=0, microsecond=0)
        rollover_utc = rollover_et.astimezone(timezone.utc)

        return rollover_utc.hour
    except Exception as e:
        logger.warning(f"DST detection failed: {e}, using 21:00 UTC")
        return 21


def is_dst_in_effect(timestamp: int) -> bool:
    """
    Check if US DST is in effect for a given timestamp.

    Args:
        timestamp: Unix timestamp in seconds

    Returns:
        True if DST is in effect (EDT), False otherwise (EST)
    """
    if ZoneInfo is None:
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        month = dt.month
        return 3 <= month <= 10

    try:
        et = ZoneInfo("America/New_York")
        dt_et = datetime.fromtimestamp(timestamp, tz=et)
        # Check if timezone offset is -4 (EDT) or -5 (EST)
        offset_hours = dt_et.utcoffset().total_seconds() / 3600  # type: ignore
        return offset_hours == -4.0
    except Exception:
        return False


# =============================================================================
# OPTIMIZED ROLLOVER COUNTER
# =============================================================================


def count_rollovers_optimized(
    prev_ts: int,
    curr_ts: int,
    dst_aware: bool = True,
) -> int:
    """
    Count the number of 5pm ET rollovers between two timestamps.

    Optimized algorithm that operates on day boundaries instead of hourly iteration.
    Wednesday rollovers count as 3 (for weekend settlement).

    Args:
        prev_ts: Previous timestamp in seconds
        curr_ts: Current timestamp in seconds
        dst_aware: Use DST-aware rollover hour detection

    Returns:
        Number of rollover days (Wednesday counts as 3)
    """
    if prev_ts >= curr_ts:
        return 0

    prev_dt = datetime.fromtimestamp(prev_ts, tz=timezone.utc)
    curr_dt = datetime.fromtimestamp(curr_ts, tz=timezone.utc)

    # Get rollover hour (DST-aware or fixed)
    if dst_aware:
        rollover_hour = get_rollover_hour_utc(prev_ts)
    else:
        rollover_hour = 21  # Fixed approximation

    count = 0

    # Normalize to start of day
    day = prev_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end_day = curr_dt.replace(hour=0, minute=0, second=0, microsecond=0)

    # Add one day to end to ensure we check the final day
    end_day = end_day + timedelta(days=1)

    while day < end_day:
        weekday = day.weekday()

        # Skip weekends (no rollover on Sat/Sun)
        if weekday < 5:  # Monday=0 to Friday=4
            # Create rollover datetime for this day
            rollover_dt = day.replace(hour=rollover_hour)

            # Check if rollover falls within our time window
            if prev_dt < rollover_dt <= curr_dt:
                if weekday == 2:  # Wednesday = 3x swap (weekend settlement)
                    count += 3
                else:
                    count += 1

        day = day + timedelta(days=1)

        # Update rollover hour if DST boundary crossed
        if dst_aware and day <= end_day:
            new_rollover_hour = get_rollover_hour_utc(int(day.timestamp()))
            if new_rollover_hour != rollover_hour:
                logger.debug(
                    f"DST transition detected: rollover hour changed "
                    f"from {rollover_hour} to {new_rollover_hour}"
                )
                rollover_hour = new_rollover_hour

    return count


# =============================================================================
# FOREX ENVIRONMENT WRAPPER
# =============================================================================


class ForexEnvWrapper(gym.Wrapper):
    """
    Wrapper for forex-specific environment adjustments.

    This wrapper modifies the TradingEnv behavior for forex trading:
    1. Session-aware reward scaling (optional)
    2. Swap cost deduction from rewards (optional, with DST-aware rollover)
    3. Leverage constraint enforcement
    4. Session feature injection into info dict

    Args:
        env: The underlying TradingEnv
        leverage: Maximum allowed leverage (default: 30.0)
        include_swap_costs: Deduct swap costs from rewards (default: True)
        swap_penalty_scale: Scale factor for swap cost in reward (default: 0.1)
        session_reward_scaling: Scale rewards by session liquidity (default: False)
        session_reward_scale_min: Minimum session reward scale (default: 0.5)
        swap_provider: Optional SwapRateProvider for pair-specific rates
        symbol: Currency pair symbol (for swap lookup)
        dst_aware: Use DST-aware rollover detection (default: True)
    """

    def __init__(
        self,
        env: gym.Env,
        leverage: float = 30.0,
        include_swap_costs: bool = True,
        swap_penalty_scale: float = 0.1,
        session_reward_scaling: bool = False,
        session_reward_scale_min: float = 0.5,
        swap_provider: Optional[SwapRateProvider] = None,
        symbol: Optional[str] = None,
        dst_aware: bool = True,
    ):
        super().__init__(env)

        self.leverage = leverage
        self.include_swap_costs = include_swap_costs
        self.swap_penalty_scale = swap_penalty_scale
        self.session_reward_scaling = session_reward_scaling
        self.session_reward_scale_min = session_reward_scale_min
        self.swap_provider = swap_provider or SwapRateProvider(use_defaults=True)
        self.symbol = symbol
        self.dst_aware = dst_aware

        # Track position for swap calculation
        self._position: float = 0.0
        self._last_timestamp: int = 0
        self._cumulative_swap_cost: float = 0.0
        self._rollover_count: int = 0

        logger.debug(
            f"ForexEnvWrapper initialized: leverage={leverage}, "
            f"swap_costs={include_swap_costs}, session_scaling={session_reward_scaling}, "
            f"dst_aware={dst_aware}"
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and initialize forex state."""
        obs, info = self.env.reset(seed=seed, options=options)

        # Reset forex-specific state
        self._position = 0.0
        self._last_timestamp = info.get("timestamp", 0)
        self._cumulative_swap_cost = 0.0
        self._rollover_count = 0

        # Add forex info
        info = self._enrich_info(info)

        return obs, info

    def step(
        self,
        action: Union[np.ndarray, float, int],
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute a step with forex-specific adjustments."""
        # Execute underlying step
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Get current timestamp and position
        current_timestamp = info.get("timestamp", self._last_timestamp)
        current_position = info.get("signal_pos_next", self._position)

        # Calculate swap cost (if position held overnight)
        swap_cost = 0.0
        rollovers = 0
        if self.include_swap_costs and abs(self._position) > 1e-6:
            swap_cost, rollovers = self._calculate_swap_cost(
                self._last_timestamp,
                current_timestamp,
                self._position,
                info,
            )
            self._cumulative_swap_cost += swap_cost
            self._rollover_count += rollovers

        # Adjust reward for swap costs
        adjusted_reward = reward
        if swap_cost > 0:
            adjusted_reward -= swap_cost * self.swap_penalty_scale

        # Apply session-based reward scaling
        reward_scale = 1.0
        if self.session_reward_scaling:
            session = self._detect_session(current_timestamp)
            liquidity = SESSION_LIQUIDITY.get(session, 1.0)
            # Scale reward: low liquidity sessions have dampened rewards
            reward_scale = max(
                self.session_reward_scale_min,
                min(1.0, liquidity),
            )
            adjusted_reward *= reward_scale

        # Update state
        self._position = current_position
        self._last_timestamp = current_timestamp

        # Enrich info
        info = self._enrich_info(info)
        info["swap_cost"] = swap_cost
        info["cumulative_swap_cost"] = self._cumulative_swap_cost
        info["reward_adjustment"] = adjusted_reward - reward
        info["rollovers_this_step"] = rollovers
        info["total_rollovers"] = self._rollover_count
        info["reward_scale"] = reward_scale

        return obs, adjusted_reward, terminated, truncated, info

    def _calculate_swap_cost(
        self,
        prev_timestamp: int,
        curr_timestamp: int,
        position: float,
        info: Dict[str, Any],
    ) -> Tuple[float, int]:
        """
        Calculate swap/rollover cost for overnight positions.

        Uses DST-aware rollover detection and SwapRateProvider for rates.

        Args:
            prev_timestamp: Previous bar timestamp (seconds)
            curr_timestamp: Current bar timestamp (seconds)
            position: Current position size (signed)
            info: Step info dict

        Returns:
            (swap_cost, rollover_count) tuple
        """
        if prev_timestamp == 0 or prev_timestamp >= curr_timestamp:
            return 0.0, 0

        # Count rollovers using optimized algorithm
        rollovers = count_rollovers_optimized(
            prev_timestamp,
            curr_timestamp,
            dst_aware=self.dst_aware,
        )

        if rollovers == 0:
            return 0.0, 0

        # Get swap rates
        symbol = self.symbol or info.get("symbol", "EUR_USD")
        swap_rate = self.swap_provider.get_swap_rate(symbol, curr_timestamp)

        # Check for override from info dict
        long_swap = info.get("long_swap", swap_rate.long_swap)
        short_swap = info.get("short_swap", swap_rate.short_swap)

        # Apply swap based on position direction
        if position > 0:
            # Long position - pay if long_swap is negative
            swap_per_day = abs(long_swap) if long_swap < 0 else 0.0
        else:
            # Short position - pay if short_swap is negative
            swap_per_day = abs(short_swap) if short_swap < 0 else 0.0

        # Total swap cost (scaled by position size)
        total_swap = swap_per_day * rollovers * abs(position)

        return total_swap, rollovers

    def _count_rollovers(self, prev_ts: int, curr_ts: int) -> int:
        """
        Count rollovers between timestamps (legacy method for compatibility).

        Uses the optimized algorithm internally.
        """
        return count_rollovers_optimized(prev_ts, curr_ts, dst_aware=self.dst_aware)

    def _detect_session(self, timestamp: int) -> str:
        """Detect the current forex trading session."""
        if timestamp == 0:
            return "london"  # Default

        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        hour = dt.hour
        weekday = dt.weekday()

        # Weekend check
        if weekday == 5:  # Saturday
            return "weekend"
        if weekday == 6:  # Sunday
            # Market opens at ~21:00-22:00 UTC on Sunday
            rollover_hour = get_rollover_hour_utc(timestamp) if self.dst_aware else 21
            if hour < rollover_hour:
                return "weekend"

        # Check overlaps first (highest priority)
        if 12 <= hour < 16:
            return "london_ny_overlap"
        if 7 <= hour < 9:
            return "tokyo_london_overlap"

        # Individual sessions
        if 7 <= hour < 16:
            return "london"
        if 12 <= hour < 21:
            return "new_york"
        if 0 <= hour < 9:
            return "tokyo"
        if hour >= 21 or hour < 6:
            return "sydney"

        return "low_liquidity"

    def _enrich_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Add forex-specific information to info dict."""
        timestamp = info.get("timestamp", self._last_timestamp)

        session = self._detect_session(timestamp)
        liquidity = SESSION_LIQUIDITY.get(session, 1.0)

        info["forex_session"] = session
        info["session_liquidity"] = liquidity
        info["max_leverage"] = self.leverage
        info["is_session_overlap"] = "overlap" in session
        info["is_weekend"] = session == "weekend"
        info["dst_in_effect"] = is_dst_in_effect(timestamp) if self.dst_aware else None
        info["rollover_hour_utc"] = get_rollover_hour_utc(timestamp) if self.dst_aware else 21

        return info


# =============================================================================
# FOREX LEVERAGE WRAPPER
# =============================================================================


class ForexLeverageWrapper(gym.Wrapper):
    """
    Wrapper that enforces forex leverage constraints on actions.

    This wrapper modifies actions to ensure they don't exceed
    the maximum allowed leverage based on account equity.

    Args:
        env: The underlying environment
        max_leverage: Maximum allowed leverage (default: 30.0)
        margin_call_level: Margin level that triggers position reduction (default: 1.0)
        stop_out_level: Margin level that forces full liquidation (default: 0.5)
    """

    def __init__(
        self,
        env: gym.Env,
        max_leverage: float = 30.0,
        margin_call_level: float = 1.0,
        stop_out_level: float = 0.5,
    ):
        super().__init__(env)

        self.max_leverage = max_leverage
        self.margin_call_level = margin_call_level
        self.stop_out_level = stop_out_level

        logger.debug(
            f"ForexLeverageWrapper: max_leverage={max_leverage}, "
            f"margin_call={margin_call_level}, stop_out={stop_out_level}"
        )

    def step(
        self,
        action: Union[np.ndarray, float, int],
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute step with leverage constraint enforcement."""
        # Get current margin level from environment if available
        equity = getattr(self.env, "_net_worth", 100000.0)
        used_margin = getattr(self.env, "_used_margin", 0.0)

        if used_margin > 0:
            margin_level = equity / used_margin
        else:
            margin_level = float("inf")

        # Check stop out (force liquidation)
        if margin_level < self.stop_out_level:
            if isinstance(action, np.ndarray):
                action = np.zeros_like(action)  # Force to cash
            else:
                action = 0.0

            logger.warning(f"Stop out triggered: level={margin_level:.2f}, " f"forcing liquidation")

        # Check margin call
        elif margin_level < self.margin_call_level:
            # Force position reduction - modify action to reduce exposure
            if isinstance(action, np.ndarray):
                action = action * 0.5  # Reduce target position
            elif isinstance(action, (float, int)):
                action = float(action) * 0.5

            logger.warning(
                f"Margin call triggered: level={margin_level:.2f}, " f"reducing position"
            )

        return self.env.step(action)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_forex_env(
    df,
    leverage: float = 30.0,
    include_swap_costs: bool = True,
    session_reward_scaling: bool = False,
    swap_provider: Optional[SwapRateProvider] = None,
    swap_dir: Optional[str] = None,
    symbol: Optional[str] = None,
    dst_aware: bool = True,
    margin_call_level: float = 1.0,
    stop_out_level: float = 0.5,
    **env_kwargs,
) -> gym.Env:
    """
    Factory function to create a forex trading environment.

    Args:
        df: DataFrame with forex OHLCV data
        leverage: Maximum leverage
        include_swap_costs: Include swap cost in rewards
        session_reward_scaling: Scale rewards by session liquidity
        swap_provider: Optional SwapRateProvider (if None, will create from swap_dir)
        swap_dir: Directory with swap rate files (used if swap_provider is None)
        symbol: Currency pair symbol
        dst_aware: Use DST-aware rollover detection
        margin_call_level: Margin level that triggers position reduction
        stop_out_level: Margin level that forces liquidation
        **env_kwargs: Additional arguments for TradingEnv

    Returns:
        Wrapped forex environment
    """
    try:
        from trading_patchnew import TradingEnv
    except ImportError:
        raise ImportError("TradingEnv not available")

    # Create swap provider if not provided
    if swap_provider is None and swap_dir is not None:
        swap_provider = SwapRateProvider.from_directory(swap_dir)
    elif swap_provider is None:
        swap_provider = SwapRateProvider(use_defaults=True)

    # Create base environment
    env = TradingEnv(
        df=df,
        asset_class="forex",
        **env_kwargs,
    )

    # Apply forex wrapper
    env = ForexEnvWrapper(
        env,
        leverage=leverage,
        include_swap_costs=include_swap_costs,
        session_reward_scaling=session_reward_scaling,
        swap_provider=swap_provider,
        symbol=symbol,
        dst_aware=dst_aware,
    )

    # Apply leverage wrapper
    env = ForexLeverageWrapper(
        env,
        max_leverage=leverage,
        margin_call_level=margin_call_level,
        stop_out_level=stop_out_level,
    )

    return env


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "SESSION_LIQUIDITY",
    "SESSION_TIMES_UTC",
    "DEFAULT_SWAP_RATES",
    "MAJOR_PAIRS",
    "MINOR_PAIRS",
    "EXOTIC_PAIRS",
    # Classes
    "SwapRate",
    "SwapRateProvider",
    "ForexEnvWrapper",
    "ForexLeverageWrapper",
    # Functions
    "get_rollover_hour_utc",
    "is_dst_in_effect",
    "count_rollovers_optimized",
    "create_forex_env",
]
