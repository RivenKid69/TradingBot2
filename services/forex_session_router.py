# -*- coding: utf-8 -*-
"""
services/forex_session_router.py
Forex Session-Aware Order Routing

Phase 6: Forex Risk Management & Services (2025-11-30)

Routes orders based on current session and liquidity.
Similar to services/session_router.py for Alpaca extended hours.

Features:
- Session detection with DST awareness
- Spread adjustment recommendations
- Optimal execution window suggestions
- Rollover time avoidance
- Weekend gap risk management

Key Differences from Equity:
- 24/5 market (Sun 5pm ET - Fri 5pm ET)
- Multiple overlapping sessions with different characteristics
- Rollover at 5pm ET affects execution
- Weekend gaps create risk

References:
- Forex Market Hours: https://www.forexfactory.com/market
- BIS Triennial Survey: https://www.bis.org/statistics/rpfx22.htm
- Rollover Time: Standard 5pm ET
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple
from zoneinfo import ZoneInfo

from adapters.models import ForexSessionType, ForexSessionWindow, FOREX_SESSION_WINDOWS

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# US Eastern timezone
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# Rollover time (5pm ET)
ROLLOVER_HOUR_ET = 17

# Rollover keepout window (minutes before and after)
ROLLOVER_KEEPOUT_MINUTES = 30

# Weekend start/end (in ET)
WEEKEND_START_HOUR_ET = 17  # Friday 5pm ET
WEEKEND_END_HOUR_ET = 17  # Sunday 5pm ET

# Session liquidity factors (relative to London session)
SESSION_LIQUIDITY = {
    ForexSessionType.SYDNEY: 0.65,
    ForexSessionType.TOKYO: 0.75,
    ForexSessionType.LONDON: 1.10,
    ForexSessionType.NEW_YORK: 1.05,
    ForexSessionType.LONDON_NY_OVERLAP: 1.40,
    ForexSessionType.TOKYO_LONDON_OVERLAP: 0.90,
    ForexSessionType.WEEKEND: 0.0,
    ForexSessionType.OFF_HOURS: 0.40,
}

# Session spread multipliers (1.0 = tightest)
SESSION_SPREAD_MULT = {
    ForexSessionType.SYDNEY: 1.50,
    ForexSessionType.TOKYO: 1.30,
    ForexSessionType.LONDON: 1.00,
    ForexSessionType.NEW_YORK: 1.00,
    ForexSessionType.LONDON_NY_OVERLAP: 0.80,  # Tightest spreads
    ForexSessionType.TOKYO_LONDON_OVERLAP: 1.10,
    ForexSessionType.WEEKEND: float("inf"),  # No trading
    ForexSessionType.OFF_HOURS: 2.00,
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ForexSessionInfo:
    """
    Information about current forex trading session.

    Attributes:
        session: Current session type
        is_open: Whether forex market is open
        liquidity_factor: Relative liquidity (1.0 = London baseline)
        spread_multiplier: Spread adjustment multiplier
        session_start_utc: Session start time (UTC)
        session_end_utc: Session end time (UTC)
        minutes_to_rollover: Minutes until next rollover
        minutes_to_weekend: Minutes until weekend (if Friday)
        in_rollover_window: Whether in rollover keepout window
        active_sessions: All currently active sessions
    """

    session: ForexSessionType
    is_open: bool
    liquidity_factor: float = 1.0
    spread_multiplier: float = 1.0
    session_start_utc: Optional[datetime] = None
    session_end_utc: Optional[datetime] = None
    minutes_to_rollover: Optional[float] = None
    minutes_to_weekend: Optional[float] = None
    in_rollover_window: bool = False
    active_sessions: List[ForexSessionType] = field(default_factory=list)


@dataclass
class RoutingDecision:
    """
    Order routing decision.

    Attributes:
        should_submit: Whether order should be submitted now
        session: Current forex session
        liquidity_factor: Session liquidity factor
        spread_multiplier: Recommended spread adjustment
        recommended_delay_sec: Suggested delay in seconds (if should_submit=False)
        reason: Human-readable reason for decision
        warnings: List of warnings/advisories
        optimal_window_utc: Suggested optimal execution window
    """

    should_submit: bool
    session: ForexSessionType
    liquidity_factor: float = 1.0
    spread_multiplier: float = 1.0
    recommended_delay_sec: Optional[float] = None
    reason: str = ""
    warnings: List[str] = field(default_factory=list)
    optimal_window_utc: Optional[Tuple[int, int]] = None  # (start_hour, end_hour)


# =============================================================================
# Session Detection
# =============================================================================


def get_current_forex_session(
    timestamp_ms: Optional[int] = None,
) -> ForexSessionInfo:
    """
    Detect current forex trading session.

    Args:
        timestamp_ms: Timestamp in milliseconds (None = current time)

    Returns:
        ForexSessionInfo with current session details
    """
    if timestamp_ms is None:
        now_utc = datetime.now(UTC)
    else:
        now_utc = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)

    hour_utc = now_utc.hour
    day_of_week = now_utc.weekday()  # 0=Monday, 6=Sunday

    # Convert to ET for rollover check
    now_et = now_utc.astimezone(ET)

    # Check if weekend
    if _is_weekend(now_et):
        return ForexSessionInfo(
            session=ForexSessionType.WEEKEND,
            is_open=False,
            liquidity_factor=0.0,
            spread_multiplier=float("inf"),
            in_rollover_window=False,
            active_sessions=[ForexSessionType.WEEKEND],
        )

    # Find all active sessions
    active_sessions: List[ForexSessionType] = []
    best_session = ForexSessionType.OFF_HOURS
    best_liquidity = 0.0

    for window in FOREX_SESSION_WINDOWS:
        if window.contains_hour(hour_utc, day_of_week):
            active_sessions.append(window.session_type)
            if window.liquidity_factor > best_liquidity:
                best_liquidity = window.liquidity_factor
                best_session = window.session_type

    # Check rollover window
    minutes_to_rollover = _minutes_to_rollover(now_et)
    in_rollover = abs(minutes_to_rollover) < ROLLOVER_KEEPOUT_MINUTES

    # Calculate minutes to weekend (if Friday)
    minutes_to_weekend = None
    if day_of_week == 4:  # Friday
        weekend_start = now_et.replace(hour=WEEKEND_START_HOUR_ET, minute=0, second=0)
        if now_et < weekend_start:
            minutes_to_weekend = (weekend_start - now_et).total_seconds() / 60

    # Get liquidity and spread for best session
    liquidity = SESSION_LIQUIDITY.get(best_session, 0.4)
    spread_mult = SESSION_SPREAD_MULT.get(best_session, 2.0)

    # Adjust for rollover window
    if in_rollover:
        liquidity *= 0.5
        spread_mult *= 2.0

    return ForexSessionInfo(
        session=best_session,
        is_open=len(active_sessions) > 0,
        liquidity_factor=liquidity,
        spread_multiplier=spread_mult,
        minutes_to_rollover=minutes_to_rollover,
        minutes_to_weekend=minutes_to_weekend,
        in_rollover_window=in_rollover,
        active_sessions=active_sessions,
    )


def _is_weekend(dt_et: datetime) -> bool:
    """Check if datetime is in forex weekend."""
    day = dt_et.weekday()  # 0=Monday, 6=Sunday
    hour = dt_et.hour

    # Saturday (all day)
    if day == 5:
        return True

    # Sunday before 5pm ET
    if day == 6 and hour < WEEKEND_END_HOUR_ET:
        return True

    # Friday after 5pm ET
    if day == 4 and hour >= WEEKEND_START_HOUR_ET:
        return True

    return False


def _minutes_to_rollover(dt_et: datetime) -> float:
    """
    Calculate minutes until next rollover (5pm ET).

    Returns:
        Minutes to rollover (negative if just passed)
    """
    rollover_today = dt_et.replace(hour=ROLLOVER_HOUR_ET, minute=0, second=0, microsecond=0)
    diff = (rollover_today - dt_et).total_seconds() / 60
    return diff


# =============================================================================
# Forex Session Router
# =============================================================================


class ForexSessionRouter:
    """
    Session-aware order routing for forex.

    Routes orders based on:
    - Current session and liquidity
    - Rollover timing
    - Weekend proximity
    - Order size vs session volume

    Usage:
        router = ForexSessionRouter(
            avoid_rollover=True,
            min_liquidity_factor=0.5,
        )

        decision = router.get_routing_decision(
            symbol="EUR_USD",
            side="BUY",
            size_usd=500000,
        )

        if decision.should_submit:
            # Execute order
            pass
        else:
            # Delay or adjust
            print(f"Delay: {decision.reason}")

    References:
        - Forex session times
        - Rollover mechanics
    """

    def __init__(
        self,
        avoid_rollover: bool = True,
        min_liquidity_factor: float = 0.5,
        avoid_weekend_proximity: bool = True,
        weekend_keepout_hours: float = 2.0,
        large_order_threshold_usd: float = 5_000_000.0,
    ) -> None:
        """
        Initialize forex session router.

        Args:
            avoid_rollover: Whether to avoid trading during rollover
            min_liquidity_factor: Minimum liquidity to accept
            avoid_weekend_proximity: Whether to avoid trading near weekend
            weekend_keepout_hours: Hours before weekend to avoid
            large_order_threshold_usd: Size threshold for special handling
        """
        self.avoid_rollover = avoid_rollover
        self.min_liquidity_factor = min_liquidity_factor
        self.avoid_weekend_proximity = avoid_weekend_proximity
        self.weekend_keepout_hours = weekend_keepout_hours
        self.large_order_threshold_usd = large_order_threshold_usd

    @property
    def current_session(self) -> ForexSessionInfo:
        """Get current forex session info."""
        return get_current_forex_session()

    def get_routing_decision(
        self,
        symbol: str,
        side: str,
        size_usd: float,
        timestamp_ms: Optional[int] = None,
    ) -> RoutingDecision:
        """
        Get routing decision for order.

        Args:
            symbol: Currency pair (e.g., "EUR_USD")
            side: "BUY" or "SELL"
            size_usd: Order size in USD
            timestamp_ms: Optional timestamp (default: now)

        Returns:
            RoutingDecision with routing parameters
        """
        session_info = get_current_forex_session(timestamp_ms)
        warnings: List[str] = []

        # Weekend - market closed
        if session_info.session == ForexSessionType.WEEKEND:
            return RoutingDecision(
                should_submit=False,
                session=ForexSessionType.WEEKEND,
                liquidity_factor=0.0,
                spread_multiplier=float("inf"),
                recommended_delay_sec=self._seconds_to_market_open(timestamp_ms),
                reason="Forex market is closed (weekend)",
                warnings=["Market closed. Order will be queued for Sunday 5pm ET open."],
            )

        # Check rollover window
        if self.avoid_rollover and session_info.in_rollover_window:
            delay = abs(session_info.minutes_to_rollover or 0) + ROLLOVER_KEEPOUT_MINUTES
            return RoutingDecision(
                should_submit=False,
                session=session_info.session,
                liquidity_factor=session_info.liquidity_factor * 0.5,
                spread_multiplier=session_info.spread_multiplier * 2.0,
                recommended_delay_sec=delay * 60,
                reason=f"Near rollover time (5pm ET). Wait {delay:.0f} minutes.",
                warnings=[
                    "Rollover time typically has wider spreads and lower liquidity.",
                    "Swap charges are applied at rollover.",
                ],
            )

        # Check weekend proximity
        if (
            self.avoid_weekend_proximity
            and session_info.minutes_to_weekend is not None
            and session_info.minutes_to_weekend < self.weekend_keepout_hours * 60
        ):
            warnings.append(
                f"Near weekend close ({session_info.minutes_to_weekend:.0f} minutes). "
                f"Consider weekend gap risk."
            )

            # For large orders near weekend, recommend waiting
            if size_usd > self.large_order_threshold_usd:
                return RoutingDecision(
                    should_submit=False,
                    session=session_info.session,
                    liquidity_factor=session_info.liquidity_factor,
                    spread_multiplier=session_info.spread_multiplier,
                    recommended_delay_sec=None,  # Wait until Monday
                    reason="Large order near weekend. Weekend gap risk too high.",
                    warnings=warnings
                    + ["Weekend gaps can be significant. Consider waiting until Monday."],
                )

        # Check minimum liquidity
        if session_info.liquidity_factor < self.min_liquidity_factor:
            # Find optimal window
            optimal = self._find_optimal_window()
            return RoutingDecision(
                should_submit=False,
                session=session_info.session,
                liquidity_factor=session_info.liquidity_factor,
                spread_multiplier=session_info.spread_multiplier,
                recommended_delay_sec=self._seconds_to_optimal_window(),
                reason=f"Low liquidity session ({session_info.session.value}). "
                f"Recommend waiting for higher liquidity.",
                warnings=[
                    f"Current liquidity factor: {session_info.liquidity_factor:.2f}",
                    f"Optimal window: {optimal[0]}:00-{optimal[1]}:00 UTC " f"(London/NY overlap)",
                ],
                optimal_window_utc=optimal,
            )

        # Large order handling
        if size_usd > self.large_order_threshold_usd:
            warnings.append(f"Large order (${size_usd:,.0f}). Consider splitting or using TWAP.")
            # Recommend London/NY overlap for large orders
            if session_info.session not in (
                ForexSessionType.LONDON,
                ForexSessionType.NEW_YORK,
                ForexSessionType.LONDON_NY_OVERLAP,
            ):
                optimal = self._find_optimal_window()
                warnings.append(
                    f"Consider executing during London/NY overlap "
                    f"({optimal[0]}:00-{optimal[1]}:00 UTC) for best liquidity."
                )

        # Add session-specific warnings
        if session_info.session == ForexSessionType.SYDNEY:
            warnings.append(
                "Sydney session: Lower liquidity for major pairs. "
                "AUD/NZD pairs have better liquidity."
            )
        elif session_info.session == ForexSessionType.TOKYO:
            warnings.append(
                "Tokyo session: Good liquidity for JPY pairs. " "EUR/USD may have wider spreads."
            )

        # All checks passed - submit
        return RoutingDecision(
            should_submit=True,
            session=session_info.session,
            liquidity_factor=session_info.liquidity_factor,
            spread_multiplier=session_info.spread_multiplier,
            reason=f"Routing to {session_info.session.value} session",
            warnings=warnings,
        )

    def _find_optimal_window(self) -> Tuple[int, int]:
        """Find optimal execution window (UTC hours)."""
        # London/NY overlap: 12:00-16:00 UTC
        return (12, 16)

    def _seconds_to_optimal_window(self) -> float:
        """Calculate seconds until optimal execution window."""
        now_utc = datetime.now(UTC)
        hour = now_utc.hour

        # Optimal window: 12:00-16:00 UTC
        if 12 <= hour < 16:
            return 0  # Already in optimal window

        if hour < 12:
            # Wait until 12:00 UTC
            target = now_utc.replace(hour=12, minute=0, second=0, microsecond=0)
        else:
            # Wait until tomorrow 12:00 UTC
            target = (now_utc + timedelta(days=1)).replace(
                hour=12, minute=0, second=0, microsecond=0
            )

        return (target - now_utc).total_seconds()

    def _seconds_to_market_open(self, timestamp_ms: Optional[int] = None) -> float:
        """Calculate seconds until market opens (Sunday 5pm ET)."""
        if timestamp_ms is None:
            now_et = datetime.now(ET)
        else:
            now_et = datetime.fromtimestamp(timestamp_ms / 1000, tz=ET)

        day = now_et.weekday()

        if day == 5:  # Saturday
            # Next open is Sunday 5pm
            days_until_sunday = 1
            target = (now_et + timedelta(days=days_until_sunday)).replace(
                hour=17, minute=0, second=0, microsecond=0
            )
        elif day == 6 and now_et.hour < 17:  # Sunday before 5pm
            target = now_et.replace(hour=17, minute=0, second=0, microsecond=0)
        elif day == 4 and now_et.hour >= 17:  # Friday after 5pm
            # Next open is Sunday 5pm (2 days)
            target = (now_et + timedelta(days=2)).replace(
                hour=17, minute=0, second=0, microsecond=0
            )
        else:
            # Market should be open
            return 0

        return (target - now_et).total_seconds()

    def adjust_spread_for_session(
        self,
        base_spread_pips: float,
        symbol: str,
        session: Optional[ForexSessionType] = None,
    ) -> float:
        """
        Adjust spread for current session.

        Args:
            base_spread_pips: Base spread in pips
            symbol: Currency pair
            session: Session type (None = current)

        Returns:
            Adjusted spread in pips
        """
        if session is None:
            session_info = self.current_session
            session = session_info.session

        multiplier = SESSION_SPREAD_MULT.get(session, 1.0)
        return base_spread_pips * multiplier

    def get_session_volume_factor(
        self,
        symbol: str,
        session: Optional[ForexSessionType] = None,
    ) -> float:
        """
        Get volume factor for session and symbol.

        Different pairs have different volume distributions across sessions.

        Args:
            symbol: Currency pair
            session: Session type (None = current)

        Returns:
            Volume factor (relative to daily volume)
        """
        if session is None:
            session_info = self.current_session
            session = session_info.session

        # Base volume by session
        base_volume = SESSION_LIQUIDITY.get(session, 0.4)

        # Adjust by pair characteristics
        symbol_upper = symbol.upper()

        if "JPY" in symbol_upper:
            # JPY pairs more active in Tokyo
            if session == ForexSessionType.TOKYO:
                base_volume *= 1.3
            elif session == ForexSessionType.SYDNEY:
                base_volume *= 1.1
        elif "EUR" in symbol_upper or "GBP" in symbol_upper:
            # European pairs more active in London
            if session == ForexSessionType.LONDON:
                base_volume *= 1.2
        elif "AUD" in symbol_upper or "NZD" in symbol_upper:
            # Antipodean pairs more active in Sydney/Tokyo
            if session in (ForexSessionType.SYDNEY, ForexSessionType.TOKYO):
                base_volume *= 1.2

        return base_volume

    def should_wait_for_better_session(
        self,
        size_usd: float,
        current_session: Optional[ForexSessionType] = None,
    ) -> Tuple[bool, str]:
        """
        Determine if order should wait for better session.

        Args:
            size_usd: Order size in USD
            current_session: Current session (None = detect)

        Returns:
            (should_wait, reason) tuple
        """
        if current_session is None:
            session_info = self.current_session
            current_session = session_info.session

        liquidity = SESSION_LIQUIDITY.get(current_session, 0.4)

        # Small orders - execute in any session
        if size_usd < 500_000:
            return (False, "Small order - execute in current session")

        # Medium orders - prefer major sessions
        if size_usd < self.large_order_threshold_usd:
            if liquidity < 0.7:
                return (
                    True,
                    f"Medium order in low liquidity session ({current_session.value}). "
                    f"Consider waiting for London/NY session.",
                )
            return (False, "Medium order in acceptable session")

        # Large orders - prefer overlap sessions
        if current_session != ForexSessionType.LONDON_NY_OVERLAP:
            if liquidity < 1.0:
                return (
                    True,
                    f"Large order (${size_usd:,.0f}) outside optimal window. "
                    f"Recommend waiting for London/NY overlap (12:00-16:00 UTC).",
                )

        return (False, "Large order in optimal session")


# =============================================================================
# Factory Functions
# =============================================================================


def create_forex_session_router(
    config: Optional[Dict[str, Any]] = None,
) -> ForexSessionRouter:
    """
    Create forex session router with configuration.

    Args:
        config: Configuration dict

    Returns:
        Configured ForexSessionRouter
    """
    config = config or {}

    return ForexSessionRouter(
        avoid_rollover=config.get("avoid_rollover", True),
        min_liquidity_factor=config.get("min_liquidity_factor", 0.5),
        avoid_weekend_proximity=config.get("avoid_weekend_proximity", True),
        weekend_keepout_hours=config.get("weekend_keepout_hours", 2.0),
        large_order_threshold_usd=config.get("large_order_threshold_usd", 5_000_000.0),
    )


# =============================================================================
# Utility Functions
# =============================================================================


def get_next_session_start(
    target_session: ForexSessionType,
    from_timestamp_ms: Optional[int] = None,
) -> int:
    """
    Get timestamp when target session next starts.

    Args:
        target_session: Target session type
        from_timestamp_ms: Starting timestamp (None = now)

    Returns:
        Timestamp in milliseconds when session starts
    """
    if from_timestamp_ms is None:
        now_utc = datetime.now(UTC)
    else:
        now_utc = datetime.fromtimestamp(from_timestamp_ms / 1000, tz=UTC)

    # Find session window
    session_window = None
    for window in FOREX_SESSION_WINDOWS:
        if window.session_type == target_session:
            session_window = window
            break

    if session_window is None:
        return from_timestamp_ms or int(now_utc.timestamp() * 1000)

    # Calculate next start
    current_hour = now_utc.hour
    current_day = now_utc.weekday()

    # Check if session is active now
    if session_window.contains_hour(current_hour, current_day):
        # Already in session
        return from_timestamp_ms or int(now_utc.timestamp() * 1000)

    # Find next start
    start_hour = session_window.start_hour_utc

    if current_hour < start_hour:
        # Later today
        target = now_utc.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    else:
        # Tomorrow
        target = (now_utc + timedelta(days=1)).replace(
            hour=start_hour, minute=0, second=0, microsecond=0
        )

    # Ensure target day is in session days
    while target.weekday() not in session_window.days_of_week:
        target += timedelta(days=1)

    return int(target.timestamp() * 1000)


def get_session_at_time(
    timestamp_ms: int,
) -> ForexSessionType:
    """
    Get forex session for a specific timestamp.

    Args:
        timestamp_ms: Timestamp in milliseconds

    Returns:
        ForexSessionType at that time
    """
    session_info = get_current_forex_session(timestamp_ms)
    return session_info.session


def is_forex_market_open(
    timestamp_ms: Optional[int] = None,
) -> bool:
    """
    Check if forex market is open.

    Args:
        timestamp_ms: Timestamp to check (None = now)

    Returns:
        True if market is open
    """
    session_info = get_current_forex_session(timestamp_ms)
    return session_info.is_open


def get_spread_multiplier_for_time(
    timestamp_ms: Optional[int] = None,
) -> float:
    """
    Get spread multiplier for a specific time.

    Args:
        timestamp_ms: Timestamp (None = now)

    Returns:
        Spread multiplier (1.0 = tightest)
    """
    session_info = get_current_forex_session(timestamp_ms)
    return session_info.spread_multiplier


def get_liquidity_factor_for_time(
    timestamp_ms: Optional[int] = None,
) -> float:
    """
    Get liquidity factor for a specific time.

    Args:
        timestamp_ms: Timestamp (None = now)

    Returns:
        Liquidity factor (1.0 = London baseline)
    """
    session_info = get_current_forex_session(timestamp_ms)
    return session_info.liquidity_factor
