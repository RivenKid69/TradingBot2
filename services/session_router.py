# -*- coding: utf-8 -*-
"""
services/session_router.py
Session-aware order routing for US equity markets.

Phase 9: Live Trading Improvements (2025-11-27)

This module provides intelligent order routing based on market session:
- Regular hours (9:30 AM - 4:00 PM ET)
- Pre-market (4:00 AM - 9:30 AM ET)
- After-hours (4:00 PM - 8:00 PM ET)

Features:
- Session detection
- Order type validation for each session
- Spread adjustment for extended hours
- Routing decisions for optimal execution
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional, Protocol, Union

import numpy as np
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# US Eastern timezone
ET = ZoneInfo("America/New_York")


# =============================================================================
# Session Types and Constants
# =============================================================================


class TradingSession(str, Enum):
    """Trading session types."""

    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"


@dataclass
class SessionInfo:
    """Information about current trading session."""

    session: TradingSession
    is_open: bool
    session_start: Optional[datetime] = None
    session_end: Optional[datetime] = None
    time_to_close_ms: Optional[int] = None
    time_to_next_open_ms: Optional[int] = None
    is_half_day: bool = False


# Session characteristics
SESSION_CHARACTERISTICS: Dict[TradingSession, Dict[str, Any]] = {
    TradingSession.CLOSED: {
        "accepts_market_orders": False,
        "accepts_limit_orders": False,
        "typical_spread_multiplier": None,  # N/A
        "typical_volume_fraction": 0.0,
        "description": "Market closed",
    },
    TradingSession.PRE_MARKET: {
        "accepts_market_orders": False,  # Must use limit orders
        "accepts_limit_orders": True,
        "typical_spread_multiplier": 2.5,  # 2-3x regular spread
        "typical_volume_fraction": 0.05,  # ~5% of daily volume
        "description": "Pre-market session (4:00 AM - 9:30 AM ET)",
    },
    TradingSession.REGULAR: {
        "accepts_market_orders": True,
        "accepts_limit_orders": True,
        "typical_spread_multiplier": 1.0,
        "typical_volume_fraction": 0.90,  # ~90% of daily volume
        "description": "Regular trading hours (9:30 AM - 4:00 PM ET)",
    },
    TradingSession.AFTER_HOURS: {
        "accepts_market_orders": False,  # Must use limit orders
        "accepts_limit_orders": True,
        "typical_spread_multiplier": 2.0,  # 2x regular spread
        "typical_volume_fraction": 0.05,  # ~5% of daily volume
        "description": "After-hours session (4:00 PM - 8:00 PM ET)",
    },
}


# =============================================================================
# Protocols
# =============================================================================


class TradingHoursProvider(Protocol):
    """Protocol for trading hours providers."""

    def is_market_open(self, ts: int, *, session_type: Optional[str] = None) -> bool:
        """Check if market is open."""
        ...

    def next_open(self, ts: int) -> int:
        """Get next market open timestamp."""
        ...

    def next_close(self, ts: int) -> int:
        """Get next market close timestamp."""
        ...


class OrderExecutionProvider(Protocol):
    """Protocol for order execution providers."""

    def submit_order(self, order: Any) -> Any:
        """Submit order."""
        ...

    def submit_extended_hours_order(self, order: Any, session: str) -> Any:
        """Submit order with extended hours handling."""
        ...


# =============================================================================
# Session Detection
# =============================================================================


def get_current_session(
    ts_ms: Optional[int] = None,
    trading_hours: Optional[TradingHoursProvider] = None,
) -> SessionInfo:
    """
    Detect current trading session.

    Args:
        ts_ms: Timestamp in milliseconds (None = current time)
        trading_hours: Optional trading hours provider

    Returns:
        SessionInfo with current session details
    """
    if ts_ms is None:
        dt = datetime.now(ET)
        ts_ms = int(dt.timestamp() * 1000)
    else:
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=ET)

    # Check if trading day (Monday-Friday, no holidays)
    # Basic check - use trading_hours provider for full holiday support
    if dt.weekday() >= 5:  # Weekend
        return SessionInfo(
            session=TradingSession.CLOSED,
            is_open=False,
        )

    # Get time in minutes from midnight
    minutes = dt.hour * 60 + dt.minute

    # Pre-market: 4:00 AM - 9:30 AM (240 - 570 minutes)
    if 240 <= minutes < 570:
        return SessionInfo(
            session=TradingSession.PRE_MARKET,
            is_open=True,
            session_start=dt.replace(hour=4, minute=0, second=0, microsecond=0),
            session_end=dt.replace(hour=9, minute=30, second=0, microsecond=0),
            time_to_close_ms=int((570 - minutes) * 60 * 1000),
        )

    # Regular hours: 9:30 AM - 4:00 PM (570 - 960 minutes)
    if 570 <= minutes < 960:
        return SessionInfo(
            session=TradingSession.REGULAR,
            is_open=True,
            session_start=dt.replace(hour=9, minute=30, second=0, microsecond=0),
            session_end=dt.replace(hour=16, minute=0, second=0, microsecond=0),
            time_to_close_ms=int((960 - minutes) * 60 * 1000),
        )

    # After-hours: 4:00 PM - 8:00 PM (960 - 1200 minutes)
    if 960 <= minutes < 1200:
        return SessionInfo(
            session=TradingSession.AFTER_HOURS,
            is_open=True,
            session_start=dt.replace(hour=16, minute=0, second=0, microsecond=0),
            session_end=dt.replace(hour=20, minute=0, second=0, microsecond=0),
            time_to_close_ms=int((1200 - minutes) * 60 * 1000),
        )

    # Closed (overnight: 8:00 PM - 4:00 AM)
    if minutes >= 1200:
        # Time until next pre-market (4:00 AM next day)
        time_to_open = (24 * 60 - minutes + 240) * 60 * 1000
    else:
        # Time until pre-market (4:00 AM same day)
        time_to_open = (240 - minutes) * 60 * 1000

    return SessionInfo(
        session=TradingSession.CLOSED,
        is_open=False,
        time_to_next_open_ms=int(time_to_open),
    )


# =============================================================================
# Order Routing
# =============================================================================


@dataclass
class RoutingDecision:
    """Decision for how to route an order."""

    should_submit: bool
    use_extended_hours: bool
    order_type_override: Optional[str] = None
    limit_price_adjustment: float = 0.0  # In basis points
    recommended_tif: str = "DAY"
    reason: str = ""
    warnings: list = None  # type: ignore

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class SessionRouter:
    """
    Session-aware order router for US equity markets.

    Determines optimal routing based on:
    - Current session (pre-market, regular, after-hours)
    - Order type (market vs limit)
    - Size and urgency
    - Symbol liquidity

    Usage:
        router = SessionRouter(trading_hours=alpaca_hours_adapter)
        decision = router.get_routing_decision(
            symbol="AAPL",
            side="BUY",
            qty=100,
            order_type="market",
        )
        if decision.should_submit:
            # Submit order with decision parameters
            ...
    """

    def __init__(
        self,
        trading_hours: Optional[TradingHoursProvider] = None,
        allow_extended_hours: bool = True,
        extended_hours_spread_multiplier: float = 2.0,
    ) -> None:
        """
        Initialize session router.

        Args:
            trading_hours: Trading hours provider
            allow_extended_hours: Whether to allow extended hours trading
            extended_hours_spread_multiplier: Spread multiplier for extended hours
        """
        self._trading_hours = trading_hours
        self._allow_extended = allow_extended_hours
        self._spread_mult = extended_hours_spread_multiplier

    @property
    def current_session(self) -> SessionInfo:
        """Get current trading session."""
        return get_current_session(trading_hours=self._trading_hours)

    def get_routing_decision(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        *,
        limit_price: Optional[float] = None,
        urgent: bool = False,
        ts_ms: Optional[int] = None,
    ) -> RoutingDecision:
        """
        Get routing decision for an order.

        Args:
            symbol: Trading symbol
            side: Order side ("buy" or "sell")
            qty: Order quantity
            order_type: Order type ("market" or "limit")
            limit_price: Limit price (if limit order)
            urgent: Whether order is urgent
            ts_ms: Timestamp (None = current time)

        Returns:
            RoutingDecision with routing parameters
        """
        session_info = get_current_session(ts_ms)
        session_chars = SESSION_CHARACTERISTICS[session_info.session]
        warnings: list = []

        # Market closed
        if session_info.session == TradingSession.CLOSED:
            return RoutingDecision(
                should_submit=False,
                use_extended_hours=False,
                reason="Market is closed",
                warnings=["Market is currently closed. Order will be queued."],
            )

        # Extended hours
        if session_info.session in (TradingSession.PRE_MARKET, TradingSession.AFTER_HOURS):
            if not self._allow_extended:
                return RoutingDecision(
                    should_submit=False,
                    use_extended_hours=False,
                    reason="Extended hours trading disabled",
                    warnings=["Extended hours trading is disabled in configuration."],
                )

            # Market orders not allowed in extended hours
            if order_type.lower() == "market":
                if limit_price is None:
                    return RoutingDecision(
                        should_submit=False,
                        use_extended_hours=True,
                        order_type_override="limit",
                        reason="Market orders not allowed in extended hours - need limit price",
                        warnings=[
                            "Market orders are not allowed during extended hours. "
                            "Please specify a limit price."
                        ],
                    )
                else:
                    # Convert to limit order
                    warnings.append("Converting market order to limit order for extended hours.")
                    return RoutingDecision(
                        should_submit=True,
                        use_extended_hours=True,
                        order_type_override="limit",
                        recommended_tif="DAY",
                        reason="Extended hours - converted to limit order",
                        warnings=warnings,
                    )

            # Limit order in extended hours - add spread adjustment
            spread_adj = (self._spread_mult - 1.0) * 100  # Convert to bps
            warnings.append(
                f"Extended hours: spreads typically {self._spread_mult:.1f}x wider. "
                f"Consider {spread_adj:.0f} bps price adjustment."
            )

            return RoutingDecision(
                should_submit=True,
                use_extended_hours=True,
                limit_price_adjustment=spread_adj,
                recommended_tif="DAY",
                reason=f"Routing to {session_info.session.value} session",
                warnings=warnings,
            )

        # Regular hours - all order types allowed
        return RoutingDecision(
            should_submit=True,
            use_extended_hours=False,
            recommended_tif="DAY",
            reason="Regular trading hours - standard routing",
            warnings=warnings,
        )

    def adjust_limit_price_for_session(
        self,
        base_price: float,
        side: str,
        session: Optional[TradingSession] = None,
    ) -> float:
        """
        Adjust limit price for session spread characteristics.

        Args:
            base_price: Original limit price
            side: Order side ("buy" or "sell")
            session: Trading session (None = current)

        Returns:
            Adjusted limit price
        """
        if session is None:
            session = self.current_session.session

        multiplier = SESSION_CHARACTERISTICS[session]["typical_spread_multiplier"]
        if multiplier is None or multiplier == 1.0:
            return base_price

        # Calculate adjustment (half the spread widening)
        adjustment_factor = (multiplier - 1.0) / 2.0

        if side.lower() == "buy":
            # Buy: reduce price by spread adjustment (more aggressive)
            return base_price * (1 - adjustment_factor / 100)
        else:
            # Sell: increase price by spread adjustment (more aggressive)
            return base_price * (1 + adjustment_factor / 100)

    def get_session_volume_estimate(
        self,
        daily_volume: float,
        session: Optional[TradingSession] = None,
    ) -> float:
        """
        Estimate volume for a session.

        Args:
            daily_volume: Average daily volume
            session: Trading session (None = current)

        Returns:
            Estimated session volume
        """
        if session is None:
            session = self.current_session.session

        fraction = SESSION_CHARACTERISTICS[session]["typical_volume_fraction"]
        return daily_volume * fraction

    def should_wait_for_regular(
        self,
        order_qty: float,
        daily_adv: float,
        urgency: str = "normal",
    ) -> bool:
        """
        Determine if order should wait for regular hours.

        For large orders, waiting for regular hours may provide better
        execution due to higher liquidity.

        Args:
            order_qty: Order quantity
            daily_adv: Average daily volume
            urgency: Order urgency ("low", "normal", "high")

        Returns:
            True if order should wait for regular hours
        """
        session = self.current_session

        # Already in regular hours
        if session.session == TradingSession.REGULAR:
            return False

        # Urgent orders should execute now
        if urgency == "high":
            return False

        # Calculate participation rate
        session_volume = self.get_session_volume_estimate(daily_adv)
        if session_volume == 0:
            return True

        participation = order_qty / session_volume

        # If participation > 10% in extended hours, consider waiting
        if participation > 0.10:
            logger.info(
                f"Large order ({participation:.1%} of session volume) - "
                f"consider waiting for regular hours"
            )
            return urgency == "low"

        return False


# =============================================================================
# Factory Functions
# =============================================================================


def create_session_router(
    trading_hours: Optional[TradingHoursProvider] = None,
    config: Optional[Dict[str, Any]] = None,
) -> SessionRouter:
    """
    Create a session router with configuration.

    Args:
        trading_hours: Trading hours provider
        config: Optional configuration dict

    Returns:
        Configured SessionRouter
    """
    config = config or {}

    return SessionRouter(
        trading_hours=trading_hours,
        allow_extended_hours=config.get("allow_extended_hours", True),
        extended_hours_spread_multiplier=config.get("extended_hours_spread_multiplier", 2.0),
    )


def create_alpaca_session_router(
    alpaca_adapter: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
) -> SessionRouter:
    """
    Create a session router for Alpaca.

    Args:
        alpaca_adapter: AlpacaTradingHoursAdapter instance
        config: Optional configuration dict

    Returns:
        SessionRouter configured for Alpaca
    """
    return create_session_router(
        trading_hours=alpaca_adapter,
        config=config,
    )


# =============================================================================
# Extended Hours Feature Functions (Phase 6 - L3 Stock Features)
# =============================================================================


def compute_gap_from_close(
    current_price: Union[float, np.ndarray],
    previous_close: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Compute gap from previous close as percentage.

    This is the primary extended hours feature - shows pre/after-market
    price movement relative to previous regular session close.

    Args:
        current_price: Current market price (pre/after-market or open).
            Can be scalar or array.
        previous_close: Previous regular session close price.
            Can be scalar or array.

    Returns:
        Gap as percentage (e.g., 1.5 for 1.5% gap up).
        Returns array if inputs are arrays.
    """
    # Handle scalar case
    if np.isscalar(current_price) and np.isscalar(previous_close):
        if previous_close <= 0 or current_price <= 0:
            return 0.0
        return ((current_price - previous_close) / previous_close) * 100.0

    # Handle array case
    current_price = np.asarray(current_price, dtype=float)
    previous_close = np.asarray(previous_close, dtype=float)

    # Create output array
    result = np.zeros_like(current_price)

    # Create valid mask
    valid_mask = (
        (previous_close > 0)
        & (current_price > 0)
        & np.isfinite(previous_close)
        & np.isfinite(current_price)
    )

    # Compute gap where valid
    with np.errstate(divide="ignore", invalid="ignore"):
        result[valid_mask] = (
            (current_price[valid_mask] - previous_close[valid_mask]) / previous_close[valid_mask]
        ) * 100.0

    return result


def compute_session_liquidity_factor(
    session: Optional[TradingSession] = None,
    ts_ms: Optional[int] = None,
) -> float:
    """
    Compute session liquidity factor.

    Returns a multiplier indicating expected liquidity relative to
    regular trading hours (1.0 = regular hours, 0.05 = extended hours).

    Args:
        session: Trading session (None = detect from timestamp)
        ts_ms: Timestamp in milliseconds

    Returns:
        Liquidity factor (0.0 to 1.0)
    """
    if session is None:
        session_info = get_current_session(ts_ms)
        session = session_info.session

    return SESSION_CHARACTERISTICS[session]["typical_volume_fraction"]


def compute_extended_hours_spread_mult(
    session: Optional[TradingSession] = None,
    ts_ms: Optional[int] = None,
) -> float:
    """
    Compute spread multiplier for current session.

    Extended hours typically have wider spreads due to lower liquidity.

    Args:
        session: Trading session (None = detect from timestamp)
        ts_ms: Timestamp in milliseconds

    Returns:
        Spread multiplier (1.0 for regular, 2.0-2.5 for extended)
    """
    if session is None:
        session_info = get_current_session(ts_ms)
        session = session_info.session

    multiplier = SESSION_CHARACTERISTICS[session]["typical_spread_multiplier"]
    return multiplier if multiplier is not None else 1.0


def add_extended_hours_features_to_df(
    df,
    timestamp_col: str = "timestamp",
    close_col: str = "close",
) -> "pd.DataFrame":
    """
    Add extended hours features to DataFrame.

    Features added:
    - gap_from_close: Pre-market gap from previous close (%)
    - session_liquidity_factor: Expected liquidity (0.05 for extended, 1.0 for regular)
    - session_spread_mult: Expected spread multiplier

    Args:
        df: DataFrame with OHLCV data
        timestamp_col: Name of timestamp column
        close_col: Name of close price column

    Returns:
        DataFrame with extended hours features
    """
    import pandas as pd
    import numpy as np

    df = df.copy()

    # Initialize columns
    df["gap_from_close"] = 0.0
    df["session_liquidity_factor"] = 1.0
    df["session_spread_mult"] = 1.0

    # Compute gap from previous close
    if close_col in df.columns:
        prev_close = df[close_col].shift(1)
        df["gap_from_close"] = compute_gap_from_close(
            df["open"].values if "open" in df.columns else df[close_col].values,
            prev_close.values,
        )
        # Vectorized computation
        valid_mask = (prev_close > 0) & (df[close_col] > 0)
        df.loc[~valid_mask, "gap_from_close"] = 0.0

    # Compute session features if we have timestamps
    if timestamp_col in df.columns:
        try:
            for i, row in df.iterrows():
                ts = row[timestamp_col]
                if isinstance(ts, (int, float)):
                    ts_ms = int(ts * 1000) if ts < 1e12 else int(ts)
                    session_info = get_current_session(ts_ms)

                    df.at[i, "session_liquidity_factor"] = SESSION_CHARACTERISTICS[
                        session_info.session
                    ]["typical_volume_fraction"]

                    spread_mult = SESSION_CHARACTERISTICS[session_info.session][
                        "typical_spread_multiplier"
                    ]
                    df.at[i, "session_spread_mult"] = spread_mult if spread_mult else 1.0
        except Exception as e:
            logger.debug(f"Could not compute session features: {e}")

    return df


@dataclass
class ExtendedHoursFeatures:
    """Extended hours feature values for observation."""

    gap_from_close: float = 0.0
    session_liquidity_factor: float = 1.0
    session_spread_mult: float = 1.0
    is_extended_hours: bool = False
    session: TradingSession = TradingSession.REGULAR


def extract_extended_hours_features(
    current_price: float,
    previous_close: float,
    ts_ms: Optional[int] = None,
) -> ExtendedHoursFeatures:
    """
    Extract extended hours features for a single observation.

    Args:
        current_price: Current market price
        previous_close: Previous session close
        ts_ms: Current timestamp in milliseconds

    Returns:
        ExtendedHoursFeatures dataclass
    """
    session_info = get_current_session(ts_ms)
    session = session_info.session

    gap = compute_gap_from_close(current_price, previous_close)
    liquidity = SESSION_CHARACTERISTICS[session]["typical_volume_fraction"]
    spread_mult = SESSION_CHARACTERISTICS[session]["typical_spread_multiplier"]

    return ExtendedHoursFeatures(
        gap_from_close=gap,
        session_liquidity_factor=liquidity,
        session_spread_mult=spread_mult if spread_mult else 1.0,
        is_extended_hours=session in (TradingSession.PRE_MARKET, TradingSession.AFTER_HOURS),
        session=session,
    )
