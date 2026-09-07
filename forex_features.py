# -*- coding: utf-8 -*-
"""
forex_features.py
Forex-specific features for ML training (Phase 4).

Features parallel to crypto Fear&Greed and equity VIX:
1. Interest Rate Differential (Carry) - analogous to funding rate in crypto
2. Relative Strength vs DXY - analogous to RS vs SPY
3. Session indicators (one-hot)
4. Spread regime
5. COT positioning (Commitments of Traders)
6. Economic calendar proximity
7. Cross-currency momentum
8. Implied volatility (FX VIX equivalent)

All features are designed for 100% backward compatibility:
- Crypto/equity data without forex features will have validity=False
- Default values are sensible (0.0 or neutral)
- No changes to existing crypto/equity flow

Data Sources:
- Interest rates: FRED API (Federal Reserve Economic Data)
- DXY: Yahoo Finance (DX-Y.NYB)
- COT: CFTC weekly reports
- Economic calendar: OANDA Labs API / ForexFactory
- Implied vol: OANDA streaming quotes

References:
- Brunnermeier et al. (2008): "Carry Trades and Currency Crashes"
- Lustig & Verdelhan (2007): "The Cross Section of Foreign Currency Risk Premia"
- Menkhoff et al. (2012): "Currency Momentum Strategies"
- Della Corte et al. (2016): "Volatility Risk Premia and Exchange Rate Predictability"

Author: Sultan Suyunbayev
Date: 2025-11-30
Version: 1.0.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, time, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# =============================================================================
# CONSTANTS
# =============================================================================

# FRED series IDs for central bank policy rates
# Source: Federal Reserve Economic Data (FRED)
RATE_SERIES: Dict[str, str] = {
    "USD": "FEDFUNDS",       # Federal Funds Rate
    "EUR": "ECBDFR",         # ECB Deposit Facility Rate
    "GBP": "IUDSOIA",        # BOE Official Bank Rate
    "JPY": "IRSTCI01JPM156N",  # BOJ Policy Rate
    "CHF": "IRSTCI01CHM156N",  # SNB Policy Rate
    "AUD": "RBATCTR",        # RBA Cash Rate
    "CAD": "IRSTCB01CAM156N",  # BOC Policy Rate
    "NZD": "RBNZCTR",        # RBNZ Official Cash Rate
}

# Interest rate normalization parameters
# Based on historical rate ranges (1990-2024)
RATE_NORMALIZATION = {
    "historical_min": -1.0,    # SNB went negative
    "historical_max": 8.0,     # Fed in early 1990s
    "typical_range": 5.0,      # Most rates are 0-5%
}

# Trading session times (UTC)
# Source: Major forex market open/close times
FOREX_SESSIONS = {
    "sydney": (time(21, 0), time(6, 0)),    # 21:00-06:00 UTC
    "tokyo": (time(0, 0), time(9, 0)),      # 00:00-09:00 UTC
    "london": (time(7, 0), time(16, 0)),    # 07:00-16:00 UTC
    "new_york": (time(12, 0), time(21, 0)), # 12:00-21:00 UTC
}

# Session overlaps (highest liquidity)
SESSION_OVERLAPS = {
    "tokyo_london": (time(7, 0), time(9, 0)),    # 07:00-09:00 UTC
    "london_ny": (time(12, 0), time(16, 0)),     # 12:00-16:00 UTC
}

# Session liquidity multipliers (relative to average)
# Source: BIS Triennial Survey 2022
SESSION_LIQUIDITY = {
    "sydney": 0.6,
    "tokyo": 0.8,
    "london": 1.3,
    "new_york": 1.2,
    "london_ny_overlap": 1.5,
    "tokyo_london_overlap": 1.0,
    "low_liquidity": 0.4,  # No major session active
}

# Spread regime thresholds (relative to average spread)
# Source: OANDA typical spreads analysis
SPREAD_REGIME_THRESHOLDS = {
    "tight": 0.7,   # Spread < 70% of average
    "normal_high": 1.3,  # Spread < 130% of average
    "wide": 2.0,    # Spread > 200% of average (news, low liquidity)
}

# COT normalization bounds
COT_NORMALIZATION = {
    "extreme_short": -2.0,   # 2 standard deviations
    "extreme_long": 2.0,
}

# DXY (Dollar Index) reference value
DXY_REFERENCE = 100.0  # DXY is centered around 100

# Typical forex volatility (annualized)
# Source: Historical major pair volatility
TYPICAL_FX_VOL = {
    "major": 0.08,     # EUR/USD, GBP/USD typically 6-10%
    "cross": 0.10,     # EUR/GBP, etc. typically 8-12%
    "exotic": 0.15,    # Emerging market pairs 12-20%
}

# High-impact economic events
HIGH_IMPACT_EVENTS = [
    "NFP", "FOMC", "CPI", "GDP", "ISM",  # USD
    "ECB", "BOE", "BOJ", "RBA", "RBNZ", "SNB", "BOC",  # Central banks
    "Employment", "Inflation", "Retail Sales",  # Major data
]

# Default values
DEFAULT_CARRY = 0.0
DEFAULT_RS_DXY = 0.0
DEFAULT_SPREAD_ZSCORE = 0.0
DEFAULT_COT_NET = 0.5  # Neutral positioning
DEFAULT_VOL = 0.0
DEFAULT_IMPLIED_VOL = 0.0

# FX Implied Volatility Indices (CBOE)
# Source: CBOE Global Markets - Currency Volatility Indices
# These track 30-day implied volatility from currency options
FX_VOL_INDICES = {
    "EUR": "^EVZ",      # CBOE EuroCurrency ETF Volatility Index (EUR/USD implied vol)
    "GBP": "^BPVIX",    # CBOE British Pound Volatility Index (not always available)
    "JPY": "^JYVIX",    # CBOE Japanese Yen Volatility Index (not always available)
    "DEFAULT": "^EVZ",  # Use EUR vol as proxy for other pairs
}

# FX VIX regime thresholds (similar to equity VIX)
# Based on historical EVZ distribution
FX_VOL_REGIMES = {
    "low": 6.0,         # Below 6% = low vol environment
    "normal": 10.0,     # 6-10% = normal
    "elevated": 15.0,   # 10-15% = elevated
    "high": 20.0,       # Above 15% = high vol (risk-off)
}

# Typical implied vol levels for reference
TYPICAL_IMPLIED_VOL = {
    "EUR_USD": 8.0,     # EUR/USD typically 6-10%
    "GBP_USD": 10.0,    # GBP/USD typically 8-12%
    "USD_JPY": 9.0,     # USD/JPY typically 7-11%
    "USD_CHF": 8.0,     # Similar to EUR/USD
    "AUD_USD": 11.0,    # Higher vol commodity currency
    "USD_CAD": 7.0,     # Lower vol, correlated to oil
    "NZD_USD": 12.0,    # Highest vol among majors
}


# =============================================================================
# ENUMS
# =============================================================================

class ForexSession(Enum):
    """Forex trading session classification."""
    SYDNEY = "sydney"
    TOKYO = "tokyo"
    LONDON = "london"
    NEW_YORK = "new_york"
    TOKYO_LONDON_OVERLAP = "tokyo_london_overlap"
    LONDON_NY_OVERLAP = "london_ny_overlap"
    LOW_LIQUIDITY = "low_liquidity"


class SpreadRegime(Enum):
    """Spread regime classification."""
    TIGHT = "tight"
    NORMAL = "normal"
    WIDE = "wide"
    EXTREME = "extreme"


class CarryRegime(Enum):
    """Carry trade regime classification."""
    NEGATIVE = -1      # Short carry (pay to hold)
    NEUTRAL = 0        # Near-zero carry
    POSITIVE = 1       # Positive carry (receive)
    HIGH_CARRY = 2     # High positive carry (>3%)


class COTPositioning(Enum):
    """COT speculator positioning classification."""
    EXTREME_SHORT = -2
    SHORT = -1
    NEUTRAL = 0
    LONG = 1
    EXTREME_LONG = 2


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ForexFeatures:
    """
    Container for forex-specific features.

    All features include validity flags to distinguish missing data from zero values,
    following the pattern established by Fear & Greed in crypto and VIX in equities.
    """
    # Interest rate differential (carry)
    base_rate: float = 0.0
    quote_rate: float = 0.0
    rate_differential: float = 0.0  # Annual % (base - quote)
    rate_differential_norm: float = 0.0  # Normalized [-1, 1]
    carry_regime: float = 0.0  # Regime indicator
    carry_valid: bool = False

    # DXY relative strength
    dxy_value: float = DXY_REFERENCE
    dxy_return_1d: float = 0.0
    dxy_return_5d: float = 0.0
    rs_vs_dxy_20d: float = 0.0  # Pair's RS vs DXY
    dxy_valid: bool = False

    # Session indicators (one-hot style)
    is_sydney: bool = False
    is_tokyo: bool = False
    is_london: bool = False
    is_new_york: bool = False
    is_overlap: bool = False
    session_liquidity: float = 1.0
    session_valid: bool = False

    # Spread dynamics
    spread_pips: float = 1.0
    spread_zscore: float = 0.0
    spread_regime: float = 0.5  # Normalized 0-1 (0=tight, 1=wide)
    spread_valid: bool = False

    # COT positioning (weekly)
    cot_net_long_pct: float = 0.5  # Normalized [0, 1]
    cot_zscore: float = 0.0
    cot_change_1w: float = 0.0
    cot_valid: bool = False

    # Economic calendar
    hours_to_next_event: float = 999.0
    next_event_impact: float = 0.0  # 0-3 scale
    is_news_window: bool = False
    calendar_valid: bool = False

    # Volatility
    realized_vol_5d: float = 0.0
    realized_vol_20d: float = 0.0
    vol_ratio: float = 1.0  # 5d / 20d
    implied_vol: float = 0.0
    vol_valid: bool = False

    # Cross-currency momentum
    cross_momentum: float = 0.0
    cross_momentum_valid: bool = False


@dataclass
class BenchmarkForexData:
    """
    Container for forex benchmark data (DXY, rates, implied vol).

    Used to pass benchmark data to feature calculators.
    """
    dxy_prices: List[float] = field(default_factory=list)
    interest_rates: Dict[str, float] = field(default_factory=dict)  # Currency -> rate
    historical_rates: Optional[pd.DataFrame] = None
    cot_data: Optional[pd.DataFrame] = None
    calendar_events: List[Dict[str, Any]] = field(default_factory=list)
    # FX Implied Volatility (from CBOE FX VIX indices like EVZ)
    implied_vol_prices: List[float] = field(default_factory=list)


@dataclass
class ForexFeatureConfig:
    """Configuration for forex feature calculation."""

    # Rate differential
    rate_normalization_scale: float = 5.0  # % points for saturation

    # Spread
    spread_lookback: int = 100  # Bars for rolling stats
    spread_zscore_clip: float = 3.0

    # COT
    cot_zscore_lookback: int = 52  # Weeks

    # Volatility
    vol_window_short: int = 5
    vol_window_long: int = 20
    annualization_factor: float = 252.0  # Trading days

    # Calendar
    news_window_hours: float = 2.0  # Hours before/after high-impact

    # Session
    session_check_enabled: bool = True


# =============================================================================
# CARRY / INTEREST RATE FEATURES
# =============================================================================

def calculate_carry_features(
    base_currency: str,
    quote_currency: str,
    rates_data: Union[pd.DataFrame, Dict[str, float]],
    timestamp_ms: Optional[int] = None,
    config: Optional[ForexFeatureConfig] = None,
) -> Tuple[float, float, float, float, bool]:
    """
    Calculate interest rate differential (carry) features.

    Carry trade logic:
    - Long position: receive base rate, pay quote rate
    - Positive differential → positive carry
    - Negative differential → negative carry (cost to hold)

    The carry trade is one of the most studied FX strategies:
    - Historically profitable but with occasional crashes
    - Related to risk appetite and volatility

    Args:
        base_currency: Base currency code (e.g., "EUR")
        quote_currency: Quote currency code (e.g., "USD")
        rates_data: DataFrame with rate columns or dict of current rates
        timestamp_ms: Current timestamp in milliseconds (optional for DataFrame lookup)
        config: Feature configuration

    Returns:
        Tuple of (base_rate, quote_rate, differential, normalized_diff, valid)

    References:
        - Brunnermeier et al. (2008): "Carry Trades and Currency Crashes"
        - Lustig & Verdelhan (2007): Cross-section of currency risk premia
    """
    if config is None:
        config = ForexFeatureConfig()

    base_rate: float = 0.0
    quote_rate: float = 0.0

    if isinstance(rates_data, dict):
        # Direct rate lookup
        base_rate = rates_data.get(base_currency, 0.0)
        quote_rate = rates_data.get(quote_currency, 0.0)

        if base_rate == 0.0 and quote_rate == 0.0:
            return (0.0, 0.0, 0.0, 0.0, False)

    elif isinstance(rates_data, pd.DataFrame):
        # DataFrame lookup (for historical data)
        base_col = f"{base_currency}_RATE"
        quote_col = f"{quote_currency}_RATE"

        if base_col not in rates_data.columns or quote_col not in rates_data.columns:
            # Try alternative column names
            for alt_suffix in ["_rate", "_Rate", ""]:
                base_alt = f"{base_currency}{alt_suffix}"
                quote_alt = f"{quote_currency}{alt_suffix}"
                if base_alt in rates_data.columns and quote_alt in rates_data.columns:
                    base_col = base_alt
                    quote_col = quote_alt
                    break
            else:
                return (0.0, 0.0, 0.0, 0.0, False)

        if timestamp_ms is not None:
            # Get most recent rate before timestamp
            ts_dt = pd.Timestamp(timestamp_ms, unit='ms', tz='UTC')

            # Handle index type
            if isinstance(rates_data.index, pd.DatetimeIndex):
                mask = rates_data.index <= ts_dt
            elif 'timestamp' in rates_data.columns:
                mask = rates_data['timestamp'] <= timestamp_ms
            else:
                # Fall back to last row
                mask = pd.Series([True] * len(rates_data))

            if not mask.any():
                return (0.0, 0.0, 0.0, 0.0, False)

            latest = rates_data.loc[mask].iloc[-1]
        else:
            # Use last row
            if len(rates_data) == 0:
                return (0.0, 0.0, 0.0, 0.0, False)
            latest = rates_data.iloc[-1]

        base_rate = float(latest.get(base_col, 0.0) if hasattr(latest, 'get') else latest[base_col])
        quote_rate = float(latest.get(quote_col, 0.0) if hasattr(latest, 'get') else latest[quote_col])
    else:
        return (0.0, 0.0, 0.0, 0.0, False)

    # Validate rates
    if not (math.isfinite(base_rate) and math.isfinite(quote_rate)):
        return (0.0, 0.0, 0.0, 0.0, False)

    # Calculate differential
    differential = base_rate - quote_rate

    # Normalize using tanh
    # ±5% differential → ±0.96 (near saturation)
    scale = config.rate_normalization_scale
    normalized = math.tanh(differential / scale)

    return (base_rate, quote_rate, differential, normalized, True)


def classify_carry_regime(differential: float) -> Tuple[CarryRegime, float]:
    """
    Classify carry trade regime from rate differential.

    Args:
        differential: Rate differential in percentage points

    Returns:
        Tuple of (CarryRegime, regime_value)
        - regime_value: -1 (negative carry) to +1 (high positive carry)
    """
    if differential < -0.5:
        return CarryRegime.NEGATIVE, -1.0
    elif differential < 0.5:
        return CarryRegime.NEUTRAL, 0.0
    elif differential < 3.0:
        return CarryRegime.POSITIVE, min(1.0, differential / 3.0)
    else:
        return CarryRegime.HIGH_CARRY, 1.0


# =============================================================================
# SESSION FEATURES
# =============================================================================

def detect_forex_session(timestamp: Union[int, datetime, pd.Timestamp]) -> ForexSession:
    """
    Detect active forex trading session from timestamp.

    Forex market has 4 major sessions with overlapping hours.
    Liquidity and spread vary significantly by session.

    Args:
        timestamp: Timestamp (ms, datetime, or pd.Timestamp)

    Returns:
        ForexSession enum indicating active session

    Note:
        Times are in UTC. DST adjustments should be handled by caller.
    """
    # Convert to datetime if needed
    if isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(timestamp / 1000.0, tz=timezone.utc)
    elif isinstance(timestamp, pd.Timestamp):
        dt = timestamp.to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = timestamp
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

    current_time = dt.time()

    # Check overlaps first (highest priority)
    for overlap_name, (start, end) in SESSION_OVERLAPS.items():
        if _time_in_range(current_time, start, end):
            if overlap_name == "london_ny":
                return ForexSession.LONDON_NY_OVERLAP
            elif overlap_name == "tokyo_london":
                return ForexSession.TOKYO_LONDON_OVERLAP

    # Check individual sessions
    for session_name, (start, end) in FOREX_SESSIONS.items():
        if _time_in_range(current_time, start, end):
            return ForexSession[session_name.upper()]

    return ForexSession.LOW_LIQUIDITY


def _time_in_range(check_time: time, start: time, end: time) -> bool:
    """Check if time is in range, handling overnight ranges."""
    if start <= end:
        return start <= check_time <= end
    else:
        # Overnight range (e.g., 21:00-06:00)
        return check_time >= start or check_time <= end


def get_session_liquidity(session: ForexSession) -> float:
    """Get liquidity multiplier for session."""
    session_key = session.value
    return SESSION_LIQUIDITY.get(session_key, 1.0)


def get_session_features(
    timestamp: Union[int, datetime, pd.Timestamp],
) -> Tuple[bool, bool, bool, bool, bool, float, ForexSession]:
    """
    Get all session-related features.

    Args:
        timestamp: Current timestamp

    Returns:
        Tuple of (is_sydney, is_tokyo, is_london, is_new_york,
                  is_overlap, liquidity, session)
    """
    session = detect_forex_session(timestamp)
    liquidity = get_session_liquidity(session)

    is_sydney = session == ForexSession.SYDNEY
    is_tokyo = session == ForexSession.TOKYO
    is_london = session == ForexSession.LONDON
    is_new_york = session == ForexSession.NEW_YORK
    is_overlap = session in (ForexSession.LONDON_NY_OVERLAP,
                             ForexSession.TOKYO_LONDON_OVERLAP)

    return (is_sydney, is_tokyo, is_london, is_new_york,
            is_overlap, liquidity, session)


# =============================================================================
# DXY RELATIVE STRENGTH FEATURES
# =============================================================================

def calculate_dxy_features(
    pair_prices: List[float],
    dxy_prices: List[float],
    window: int = 20,
) -> Tuple[float, float, float, bool]:
    """
    Calculate DXY-related features including relative strength.

    DXY (US Dollar Index) tracks USD against basket of 6 currencies.
    Components: EUR (57.6%), JPY (13.6%), GBP (11.9%), CAD (9.1%),
                SEK (4.2%), CHF (3.6%)

    For USD-based pairs:
    - Pair RS < DXY RS → pair underperforming vs USD basket
    - Pair RS > DXY RS → pair outperforming vs USD basket

    Args:
        pair_prices: Currency pair close prices (most recent last)
        dxy_prices: DXY close prices (most recent last)
        window: Lookback period for RS calculation

    Returns:
        Tuple of (dxy_return_5d, rs_vs_dxy, dxy_value, valid)
    """
    if not pair_prices or not dxy_prices:
        return (0.0, 0.0, DXY_REFERENCE, False)

    if len(dxy_prices) < window + 1 or len(pair_prices) < window + 1:
        # Try shorter calculation with available data
        min_len = min(len(pair_prices), len(dxy_prices))
        if min_len < 2:
            return (0.0, 0.0, dxy_prices[-1] if dxy_prices else DXY_REFERENCE, False)
        window = min_len - 1

    try:
        # Get current DXY
        dxy_now = float(dxy_prices[-1])

        if not math.isfinite(dxy_now) or dxy_now <= 0:
            return (0.0, 0.0, DXY_REFERENCE, False)

        # 5-day return (or shorter if not enough data)
        lookback_5d = min(5, len(dxy_prices) - 1)
        dxy_past_5d = float(dxy_prices[-(lookback_5d + 1)])
        dxy_return_5d = (dxy_now / dxy_past_5d - 1.0) if dxy_past_5d > 0 else 0.0

        # RS calculation
        pair_now = float(pair_prices[-1])
        pair_past = float(pair_prices[-(window + 1)])
        dxy_past = float(dxy_prices[-(window + 1)])

        # Validate prices
        if not all(math.isfinite(p) and p > 0 for p in
                   [pair_now, pair_past, dxy_now, dxy_past]):
            return (dxy_return_5d, 0.0, dxy_now, False)

        # Calculate returns
        pair_return = (pair_now / pair_past) - 1.0
        dxy_return = (dxy_now / dxy_past) - 1.0

        # RS vs DXY
        # For XXX/USD pairs: positive RS means XXX outperforming DXY basket
        if abs(dxy_return) < 1e-8:
            raw_rs = pair_return
        else:
            raw_rs = (1.0 + pair_return) / (1.0 + dxy_return) - 1.0

        if not math.isfinite(raw_rs):
            return (dxy_return_5d, 0.0, dxy_now, False)

        # Normalize using tanh
        # ±10% RS → ±0.76
        normalized_rs = math.tanh(raw_rs * 5.0)

        return (dxy_return_5d, normalized_rs, dxy_now, True)

    except Exception:
        return (0.0, 0.0, DXY_REFERENCE, False)


def normalize_dxy_value(dxy_value: float) -> float:
    """
    Normalize DXY value for observation vector.

    DXY typically ranges from 70 to 130, centered around 100.
    Uses (dxy - 100) / 20 with tanh to bound to [-1, 1].

    Args:
        dxy_value: Raw DXY value

    Returns:
        Normalized DXY in approximately [-1, 1] range
    """
    if not math.isfinite(dxy_value):
        return 0.0

    # Center at 100, scale by 20
    normalized = math.tanh((dxy_value - 100.0) / 20.0)
    return normalized


# =============================================================================
# SPREAD FEATURES
# =============================================================================

def calculate_spread_features(
    current_spread_pips: float,
    spread_history: List[float],
    config: Optional[ForexFeatureConfig] = None,
) -> Tuple[float, float, SpreadRegime, bool]:
    """
    Calculate spread-related features.

    Spread dynamics are crucial for forex:
    - Wide spreads indicate low liquidity or high volatility
    - Spread widening before news events is common
    - Abnormal spread may indicate market stress

    Args:
        current_spread_pips: Current bid-ask spread in pips
        spread_history: Historical spread values
        config: Feature configuration

    Returns:
        Tuple of (zscore, regime_value, SpreadRegime, valid)
    """
    if config is None:
        config = ForexFeatureConfig()

    if not math.isfinite(current_spread_pips) or current_spread_pips < 0:
        return (0.0, 0.5, SpreadRegime.NORMAL, False)

    if not spread_history or len(spread_history) < 10:
        # Not enough history for statistics
        return (0.0, 0.5, SpreadRegime.NORMAL, False)

    try:
        # Use rolling lookback
        lookback = min(len(spread_history), config.spread_lookback)
        recent_spreads = spread_history[-lookback:]

        # Calculate statistics
        mean_spread = np.nanmean(recent_spreads)
        std_spread = np.nanstd(recent_spreads)

        if not (math.isfinite(mean_spread) and math.isfinite(std_spread)):
            return (0.0, 0.5, SpreadRegime.NORMAL, False)

        if std_spread < 1e-8 or mean_spread < 1e-8:
            return (0.0, 0.5, SpreadRegime.NORMAL, False)

        # Z-score
        zscore = (current_spread_pips - mean_spread) / std_spread
        zscore = max(-config.spread_zscore_clip, min(config.spread_zscore_clip, zscore))

        # Regime classification
        spread_ratio = current_spread_pips / mean_spread

        if spread_ratio < SPREAD_REGIME_THRESHOLDS["tight"]:
            regime = SpreadRegime.TIGHT
            regime_value = 0.0
        elif spread_ratio < SPREAD_REGIME_THRESHOLDS["normal_high"]:
            regime = SpreadRegime.NORMAL
            regime_value = 0.5
        elif spread_ratio < SPREAD_REGIME_THRESHOLDS["wide"]:
            regime = SpreadRegime.WIDE
            regime_value = 0.75
        else:
            regime = SpreadRegime.EXTREME
            regime_value = 1.0

        return (zscore, regime_value, regime, True)

    except Exception:
        return (0.0, 0.5, SpreadRegime.NORMAL, False)


# =============================================================================
# COT (COMMITMENTS OF TRADERS) FEATURES
# =============================================================================

def calculate_cot_features(
    currency: str,
    cot_data: pd.DataFrame,
    timestamp_ms: Optional[int] = None,
    config: Optional[ForexFeatureConfig] = None,
) -> Tuple[float, float, float, COTPositioning, bool]:
    """
    Calculate COT positioning features.

    CFTC Commitments of Traders report shows speculator positioning.
    - Net long positioning may indicate bullish sentiment
    - Extreme positioning may signal reversal risk
    - Changes in positioning indicate sentiment shifts

    Args:
        currency: Currency code
        cot_data: COT DataFrame with columns like 'EUR_NET', 'EUR_LONG', etc.
        timestamp_ms: Current timestamp
        config: Feature configuration

    Returns:
        Tuple of (net_pct, zscore, change_1w, positioning, valid)

    References:
        - CFTC weekly reports
        - Klitgaard & Weir (2004): Exchange Rate Changes and Net Positions
    """
    if config is None:
        config = ForexFeatureConfig()

    if cot_data is None or cot_data.empty:
        return (0.5, 0.0, 0.0, COTPositioning.NEUTRAL, False)

    # Column naming convention
    net_col = f"{currency}_NET"
    long_col = f"{currency}_LONG"
    short_col = f"{currency}_SHORT"
    oi_col = f"{currency}_OI"  # Open interest

    # Try alternative column names
    for suffix in ["_net", "_NET_LONG", "_NET_SPEC"]:
        if f"{currency}{suffix}" in cot_data.columns:
            net_col = f"{currency}{suffix}"
            break

    if net_col not in cot_data.columns:
        return (0.5, 0.0, 0.0, COTPositioning.NEUTRAL, False)

    try:
        # Get relevant data
        if timestamp_ms is not None:
            ts_dt = pd.Timestamp(timestamp_ms, unit='ms', tz='UTC')
            mask = cot_data.index <= ts_dt if isinstance(cot_data.index, pd.DatetimeIndex) else pd.Series([True] * len(cot_data))
            data = cot_data.loc[mask][net_col]
        else:
            data = cot_data[net_col]

        if len(data) == 0:
            return (0.5, 0.0, 0.0, COTPositioning.NEUTRAL, False)

        # Current net position
        net_position = float(data.iloc[-1])

        # Get open interest for normalization (if available)
        if oi_col in cot_data.columns:
            oi = float(cot_data[oi_col].iloc[-1])
            if oi > 0:
                net_pct = net_position / oi
            else:
                # Fallback: normalize by historical range
                net_pct = (net_position - data.min()) / (data.max() - data.min() + 1e-8)
        else:
            # Normalize by historical range
            net_pct = (net_position - data.min()) / (data.max() - data.min() + 1e-8)

        # Z-score of current position
        lookback = min(len(data), config.cot_zscore_lookback)
        recent_data = data.iloc[-lookback:]
        mean_net = recent_data.mean()
        std_net = recent_data.std()

        if std_net > 0:
            zscore = (net_position - mean_net) / std_net
            zscore = max(-3.0, min(3.0, zscore))
        else:
            zscore = 0.0

        # Week-over-week change
        if len(data) >= 2:
            change_1w = net_position - float(data.iloc[-2])
            # Normalize change
            change_1w_norm = math.tanh(change_1w / (abs(net_position) + 1e-8) * 5)
        else:
            change_1w_norm = 0.0

        # Positioning classification
        if zscore < -1.5:
            positioning = COTPositioning.EXTREME_SHORT
        elif zscore < -0.5:
            positioning = COTPositioning.SHORT
        elif zscore < 0.5:
            positioning = COTPositioning.NEUTRAL
        elif zscore < 1.5:
            positioning = COTPositioning.LONG
        else:
            positioning = COTPositioning.EXTREME_LONG

        # Bound net_pct to [0, 1]
        net_pct = max(0.0, min(1.0, net_pct))

        return (net_pct, zscore, change_1w_norm, positioning, True)

    except Exception:
        return (0.5, 0.0, 0.0, COTPositioning.NEUTRAL, False)


# =============================================================================
# VOLATILITY FEATURES
# =============================================================================

def calculate_volatility_features(
    prices: List[float],
    config: Optional[ForexFeatureConfig] = None,
) -> Tuple[float, float, float, bool]:
    """
    Calculate realized volatility features.

    Two volatility measures:
    - Short-term (5d): Recent volatility for timing
    - Long-term (20d): Baseline volatility

    Vol ratio (5d/20d) indicates volatility regime:
    - < 1: Volatility compression (may precede breakout)
    - > 1: Volatility expansion (trending/news)

    Args:
        prices: Close prices (most recent last)
        config: Feature configuration

    Returns:
        Tuple of (vol_5d, vol_20d, vol_ratio, valid)
    """
    if config is None:
        config = ForexFeatureConfig()

    if not prices or len(prices) < config.vol_window_long + 1:
        return (0.0, 0.0, 1.0, False)

    try:
        prices_arr = np.array(prices, dtype=np.float64)

        # Calculate log returns
        returns = np.diff(np.log(prices_arr))

        if len(returns) < config.vol_window_long:
            return (0.0, 0.0, 1.0, False)

        # Realized volatility (annualized)
        vol_short = np.std(returns[-config.vol_window_short:]) * np.sqrt(config.annualization_factor)
        vol_long = np.std(returns[-config.vol_window_long:]) * np.sqrt(config.annualization_factor)

        if not (math.isfinite(vol_short) and math.isfinite(vol_long)):
            return (0.0, 0.0, 1.0, False)

        # Vol ratio
        if vol_long > 1e-8:
            vol_ratio = vol_short / vol_long
        else:
            vol_ratio = 1.0

        # Normalize volatility using tanh
        # 10% annualized vol → ~0.76
        vol_short_norm = math.tanh(vol_short * 10)
        vol_long_norm = math.tanh(vol_long * 10)

        return (vol_short_norm, vol_long_norm, vol_ratio, True)

    except Exception:
        return (0.0, 0.0, 1.0, False)


# =============================================================================
# IMPLIED VOLATILITY FEATURES
# =============================================================================

def calculate_implied_vol_features(
    implied_vol_prices: List[float],
    symbol: str = "",
    window: int = 20,
) -> Tuple[float, float, str, bool]:
    """
    Calculate FX implied volatility features from CBOE FX VIX indices.

    FX implied volatility provides forward-looking risk expectations:
    - EVZ (EUR/USD vol index): Primary FX fear gauge
    - Low implied vol (<6%): Carry-friendly environment
    - High implied vol (>15%): Risk-off, potential for reversals

    Unlike realized vol, implied vol is priced from options and reflects
    market's expectation of future volatility over 30 days.

    Args:
        implied_vol_prices: Historical FX VIX values (e.g., EVZ)
        symbol: Currency pair for pair-specific typical vol reference
        window: Lookback for percentile ranking

    Returns:
        Tuple of (current_vol, vol_percentile, regime, valid)
        - current_vol: Latest FX VIX value (normalized 0-1 scale)
        - vol_percentile: Where current vol sits in historical distribution
        - regime: "low", "normal", "elevated", or "high"
        - valid: Whether calculation succeeded

    References:
        - CBOE: FX Volatility Index methodology
        - Della Corte et al. (2016): "Volatility Risk Premia and Exchange Rate Predictability"
    """
    if not implied_vol_prices or len(implied_vol_prices) < 1:
        return (0.0, 0.5, "normal", False)

    try:
        # Get current implied vol
        current_vol = float(implied_vol_prices[-1])

        if not math.isfinite(current_vol) or current_vol < 0:
            return (0.0, 0.5, "normal", False)

        # Classify regime based on FX_VOL_REGIMES thresholds
        if current_vol < FX_VOL_REGIMES["low"]:
            regime = "low"
        elif current_vol < FX_VOL_REGIMES["normal"]:
            regime = "normal"
        elif current_vol < FX_VOL_REGIMES["elevated"]:
            regime = "elevated"
        else:
            regime = "high"

        # Calculate percentile rank if enough history
        if len(implied_vol_prices) >= window:
            lookback = implied_vol_prices[-window:]
            # Percentile rank: what % of historical values are below current
            below_count = sum(1 for v in lookback if v < current_vol)
            vol_percentile = below_count / len(lookback)
        else:
            # Not enough history - use regime-based estimate
            regime_percentile = {
                "low": 0.2,
                "normal": 0.5,
                "elevated": 0.75,
                "high": 0.9,
            }
            vol_percentile = regime_percentile.get(regime, 0.5)

        # Normalize current vol to [0, 1] scale
        # Typical FX vol range: 4-25%, use 20 as saturation point
        # Reference: TYPICAL_IMPLIED_VOL values around 7-12%
        typical_vol = TYPICAL_IMPLIED_VOL.get(
            symbol.upper().replace("/", "_"),
            10.0  # Default typical vol
        )
        # Normalize: 0 at 0%, 0.5 at typical, 1.0 at 2x typical
        normalized_vol = min(1.0, current_vol / (typical_vol * 2.0))

        return (normalized_vol, vol_percentile, regime, True)

    except Exception:
        return (0.0, 0.5, "normal", False)


def get_fx_vol_regime_multiplier(regime: str) -> float:
    """
    Get liquidity/cost multiplier based on FX vol regime.

    Similar to equity VIX regimes affecting execution costs:
    - Low vol: tighter spreads, better fills
    - High vol: wider spreads, more slippage

    Args:
        regime: Vol regime ("low", "normal", "elevated", "high")

    Returns:
        Multiplier for spread/slippage adjustment (0.8 - 1.5)
    """
    regime_multipliers = {
        "low": 0.8,       # Low vol = tighter spreads
        "normal": 1.0,    # Baseline
        "elevated": 1.2,  # Slightly wider
        "high": 1.5,      # Much wider during stress
    }
    return regime_multipliers.get(regime, 1.0)


# =============================================================================
# CROSS-CURRENCY MOMENTUM FEATURES
# =============================================================================

def calculate_cross_momentum(
    pair_returns: Dict[str, float],
    target_pair: str,
) -> Tuple[float, bool]:
    """
    Calculate cross-currency momentum signal.

    Based on cross-sectional momentum in FX:
    - Currency pairs with recent outperformance tend to continue
    - Relative rank among pairs indicates strength

    Args:
        pair_returns: Dict of pair -> return for momentum period
        target_pair: The pair to calculate momentum for

    Returns:
        Tuple of (momentum_rank, valid)
        - momentum_rank: -1 (worst) to +1 (best) among pairs

    References:
        - Menkhoff et al. (2012): "Currency Momentum Strategies"
    """
    if not pair_returns or target_pair not in pair_returns:
        return (0.0, False)

    try:
        returns_list = list(pair_returns.values())
        target_return = pair_returns[target_pair]

        if len(returns_list) < 2:
            # Not enough pairs for cross-sectional ranking
            return (math.tanh(target_return * 10), True)

        # Calculate rank (0 = worst, 1 = best)
        sorted_returns = sorted(returns_list)
        rank = sorted_returns.index(target_return)

        # Normalize to [-1, 1]
        n = len(sorted_returns)
        rank_normalized = 2.0 * rank / (n - 1) - 1.0

        return (rank_normalized, True)

    except Exception:
        return (0.0, False)


# =============================================================================
# MAIN FEATURE EXTRACTION
# =============================================================================

def extract_forex_features(
    row: Any,
    symbol: str,
    benchmark_data: Optional[BenchmarkForexData] = None,
    pair_prices: Optional[List[float]] = None,
    spread_history: Optional[List[float]] = None,
    config: Optional[ForexFeatureConfig] = None,
) -> ForexFeatures:
    """
    Extract all forex-specific features from a data row.

    This is the main entry point for forex feature extraction.
    Features are designed to be analogous to crypto's Fear & Greed index
    and equity VIX features.

    Args:
        row: Data row (DataFrame row or dict) with market data
        symbol: Currency pair (e.g., "EUR_USD")
        benchmark_data: Container with DXY, rates, COT data
        pair_prices: Historical prices for the pair
        spread_history: Historical spread values
        config: Feature configuration

    Returns:
        ForexFeatures dataclass with all extracted features and validity flags

    Example:
        >>> benchmark = BenchmarkForexData(
        ...     dxy_prices=dxy_history,
        ...     interest_rates={"EUR": 4.0, "USD": 5.25},
        ... )
        >>> features = extract_forex_features(row, "EUR_USD", benchmark)
        >>> print(f"Carry: {features.rate_differential}%, Session: {features.is_london}")
    """
    if config is None:
        config = ForexFeatureConfig()

    features = ForexFeatures()

    # Parse currency pair
    currencies = _parse_currency_pair(symbol)
    if currencies is None:
        return features

    base_currency, quote_currency = currencies

    # Get timestamp from row
    timestamp = _extract_timestamp_from_row(row)

    # ===================
    # CARRY FEATURES
    # ===================
    if benchmark_data and benchmark_data.interest_rates:
        rates_data = benchmark_data.interest_rates
        if benchmark_data.historical_rates is not None:
            rates_data = benchmark_data.historical_rates

        base_rate, quote_rate, diff, diff_norm, carry_valid = calculate_carry_features(
            base_currency, quote_currency, rates_data, timestamp, config
        )

        if carry_valid:
            features.base_rate = base_rate
            features.quote_rate = quote_rate
            features.rate_differential = diff
            features.rate_differential_norm = diff_norm
            _, regime_val = classify_carry_regime(diff)
            features.carry_regime = regime_val
            features.carry_valid = True

    # ===================
    # SESSION FEATURES
    # ===================
    if timestamp is not None and config.session_check_enabled:
        (is_sydney, is_tokyo, is_london, is_new_york,
         is_overlap, liquidity, session) = get_session_features(timestamp)

        features.is_sydney = is_sydney
        features.is_tokyo = is_tokyo
        features.is_london = is_london
        features.is_new_york = is_new_york
        features.is_overlap = is_overlap
        features.session_liquidity = liquidity
        features.session_valid = True

    # ===================
    # DXY FEATURES
    # ===================
    if pair_prices and benchmark_data and benchmark_data.dxy_prices:
        dxy_ret_5d, rs_vs_dxy, dxy_val, dxy_valid = calculate_dxy_features(
            pair_prices, benchmark_data.dxy_prices, window=20
        )

        if dxy_valid:
            features.dxy_value = normalize_dxy_value(dxy_val)
            features.dxy_return_5d = dxy_ret_5d
            features.rs_vs_dxy_20d = rs_vs_dxy
            features.dxy_valid = True

    # ===================
    # SPREAD FEATURES
    # ===================
    current_spread = _extract_spread_from_row(row)
    if current_spread is not None and spread_history:
        zscore, regime_val, regime, spread_valid = calculate_spread_features(
            current_spread, spread_history, config
        )

        if spread_valid:
            features.spread_pips = current_spread
            features.spread_zscore = zscore
            features.spread_regime = regime_val
            features.spread_valid = True

    # ===================
    # COT FEATURES
    # ===================
    if benchmark_data and benchmark_data.cot_data is not None:
        net_pct, zscore, change, positioning, cot_valid = calculate_cot_features(
            base_currency, benchmark_data.cot_data, timestamp, config
        )

        if cot_valid:
            features.cot_net_long_pct = net_pct
            features.cot_zscore = zscore
            features.cot_change_1w = change
            features.cot_valid = True

    # ===================
    # VOLATILITY FEATURES
    # ===================
    if pair_prices:
        vol_short, vol_long, vol_ratio, vol_valid = calculate_volatility_features(
            pair_prices, config
        )

        if vol_valid:
            features.realized_vol_5d = vol_short
            features.realized_vol_20d = vol_long
            features.vol_ratio = vol_ratio
            features.vol_valid = True

    # ===================
    # IMPLIED VOLATILITY
    # ===================
    if benchmark_data and benchmark_data.implied_vol_prices:
        implied_vol, vol_pct, vol_regime, implied_valid = calculate_implied_vol_features(
            benchmark_data.implied_vol_prices, symbol
        )

        if implied_valid:
            features.implied_vol = implied_vol
            # Note: vol_valid already set by realized vol, but implied vol enhances it
            if not features.vol_valid:
                features.vol_valid = True

    # ===================
    # ECONOMIC CALENDAR
    # ===================
    if timestamp and benchmark_data and benchmark_data.calendar_events:
        hours_to_next, impact, is_news = _check_calendar_proximity(
            timestamp, benchmark_data.calendar_events,
            [base_currency, quote_currency], config
        )

        features.hours_to_next_event = hours_to_next
        features.next_event_impact = impact
        features.is_news_window = is_news
        features.calendar_valid = True

    return features


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _parse_currency_pair(symbol: str) -> Optional[Tuple[str, str]]:
    """Parse currency pair into base and quote currencies."""
    if not symbol:
        return None

    # Handle different formats
    symbol = symbol.upper().strip()

    # Format: EUR_USD or EURUSD
    if "_" in symbol:
        parts = symbol.split("_")
        if len(parts) == 2:
            return (parts[0], parts[1])
    elif "/" in symbol:
        parts = symbol.split("/")
        if len(parts) == 2:
            return (parts[0], parts[1])
    elif len(symbol) == 6:
        return (symbol[:3], symbol[3:])

    return None


def _extract_timestamp_from_row(row: Any) -> Optional[int]:
    """Extract timestamp from data row."""
    if row is None:
        return None

    ts_columns = ["timestamp", "ts", "time", "date", "datetime"]

    for col in ts_columns:
        try:
            if hasattr(row, "get"):
                val = row.get(col)
            else:
                val = getattr(row, col, None)

            if val is not None:
                if isinstance(val, (int, float)):
                    return int(val)
                elif isinstance(val, (datetime, pd.Timestamp)):
                    return int(val.timestamp() * 1000)
        except (TypeError, ValueError, AttributeError):
            continue

    return None


def _extract_spread_from_row(row: Any) -> Optional[float]:
    """Extract spread from data row."""
    if row is None:
        return None

    spread_columns = ["spread", "spread_pips", "bid_ask_spread", "ask_bid"]

    for col in spread_columns:
        try:
            if hasattr(row, "get"):
                val = row.get(col)
            else:
                val = getattr(row, col, None)

            if val is not None:
                val_float = float(val)
                if math.isfinite(val_float) and val_float >= 0:
                    return val_float
        except (TypeError, ValueError, AttributeError):
            continue

    # Try to calculate from bid/ask
    try:
        if hasattr(row, "get"):
            ask = row.get("ask") or row.get("ask_close") or row.get("close_ask")
            bid = row.get("bid") or row.get("bid_close") or row.get("close_bid")
        else:
            ask = getattr(row, "ask", None) or getattr(row, "ask_close", None)
            bid = getattr(row, "bid", None) or getattr(row, "bid_close", None)

        if ask is not None and bid is not None:
            spread = float(ask) - float(bid)
            if spread >= 0:
                return spread
    except (TypeError, ValueError, AttributeError):
        pass

    return None


def _check_calendar_proximity(
    timestamp_ms: int,
    events: List[Dict[str, Any]],
    currencies: List[str],
    config: ForexFeatureConfig,
) -> Tuple[float, float, bool]:
    """
    Check proximity to economic calendar events.

    Returns:
        Tuple of (hours_to_next_event, impact_level, is_in_news_window)
    """
    if not events:
        return (999.0, 0.0, False)

    current_dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)

    min_hours = 999.0
    max_impact = 0.0
    is_news_window = False

    for event in events:
        event_currency = event.get("currency", "")
        if event_currency not in currencies:
            continue

        event_time = event.get("datetime") or event.get("time") or event.get("timestamp")
        if event_time is None:
            continue

        if isinstance(event_time, (int, float)):
            event_dt = datetime.fromtimestamp(event_time / 1000.0, tz=timezone.utc)
        elif isinstance(event_time, str):
            try:
                event_dt = datetime.fromisoformat(event_time.replace("Z", "+00:00"))
            except ValueError:
                continue
        else:
            event_dt = event_time
            if event_dt.tzinfo is None:
                event_dt = event_dt.replace(tzinfo=timezone.utc)

        # Calculate hours until event
        delta = event_dt - current_dt
        hours = delta.total_seconds() / 3600.0

        # Skip past events (unless within window)
        if hours < -config.news_window_hours:
            continue

        impact = float(event.get("impact", 0))

        # Check if within news window
        if abs(hours) < config.news_window_hours and impact >= 2:
            is_news_window = True

        # Track closest future event
        if 0 <= hours < min_hours:
            min_hours = hours
            max_impact = max(max_impact, impact)

    return (min_hours, max_impact, is_news_window)


# =============================================================================
# DATAFRAME UTILITIES
# =============================================================================

def add_forex_features_to_dataframe(
    df: pd.DataFrame,
    symbol: str,
    dxy_df: Optional[pd.DataFrame] = None,
    rates_df: Optional[pd.DataFrame] = None,
    cot_df: Optional[pd.DataFrame] = None,
    calendar_events: Optional[List[Dict[str, Any]]] = None,
    implied_vol_df: Optional[pd.DataFrame] = None,
    config: Optional[ForexFeatureConfig] = None,
) -> pd.DataFrame:
    """
    Add forex-specific features to a DataFrame.

    This function adds all forex features as new columns to the input DataFrame.
    Used during data preparation for training/inference.

    Args:
        df: Main DataFrame with OHLCV data
        symbol: Currency pair (e.g., "EUR_USD")
        dxy_df: DXY price DataFrame
        rates_df: Interest rates DataFrame
        cot_df: COT positioning DataFrame
        calendar_events: Economic calendar events
        implied_vol_df: FX implied volatility DataFrame (EVZ from CBOE)
        config: Feature configuration

    Returns:
        DataFrame with added forex feature columns

    New columns added:
        - carry_diff: Interest rate differential (normalized)
        - carry_regime: Carry regime indicator
        - dxy_value: Normalized DXY value
        - dxy_rs: Relative strength vs DXY
        - session_*: Session indicator columns
        - spread_zscore: Spread z-score
        - spread_regime: Spread regime
        - cot_net: COT net positioning
        - vol_ratio: Short/long volatility ratio
        - implied_vol: FX implied volatility (from EVZ/CBOE)
    """
    if config is None:
        config = ForexFeatureConfig()

    df = df.copy()
    n_rows = len(df)

    # Initialize columns with default values
    feature_cols = {
        "carry_diff": 0.0,
        "carry_regime": 0.0,
        "dxy_value": 0.0,
        "dxy_rs": 0.0,
        "session_sydney": 0.0,
        "session_tokyo": 0.0,
        "session_london": 0.0,
        "session_new_york": 0.0,
        "session_overlap": 0.0,
        "session_liquidity": 1.0,
        "spread_zscore": 0.0,
        "spread_regime": 0.5,
        "cot_net": 0.5,
        "cot_zscore": 0.0,
        "vol_5d": 0.0,
        "vol_20d": 0.0,
        "vol_ratio": 1.0,
        "implied_vol": 0.0,
        "hours_to_event": 999.0,
        "is_news_window": 0.0,
    }

    for col, default in feature_cols.items():
        df[col] = default

    # Get close prices
    close_col = "close" if "close" in df.columns else "Close"
    if close_col not in df.columns:
        return df

    pair_prices = df[close_col].tolist()

    # Align benchmark data using timestamp-based merge
    dxy_aligned = _align_forex_benchmark(df, dxy_df, "close") if dxy_df is not None else [None] * n_rows
    implied_vol_aligned = _align_forex_benchmark(df, implied_vol_df, "close") if implied_vol_df is not None else [None] * n_rows

    # Get spread history
    spread_col = None
    for col in ["spread", "spread_pips", "bid_ask_spread"]:
        if col in df.columns:
            spread_col = col
            break

    spread_history_all = df[spread_col].tolist() if spread_col else []

    # Build interest rates dict
    currencies = _parse_currency_pair(symbol)
    interest_rates: Dict[str, float] = {}
    if rates_df is not None and currencies:
        base, quote = currencies
        for curr in [base, quote]:
            for col_suffix in ["_RATE", "_rate", ""]:
                col_name = f"{curr}{col_suffix}"
                if col_name in rates_df.columns:
                    interest_rates[curr] = float(rates_df[col_name].iloc[-1])
                    break

    # Build cumulative lists
    dxy_cumulative: List[float] = []
    implied_vol_cumulative: List[float] = []

    # Calculate rolling features
    for i in range(n_rows):
        current_prices = pair_prices[:i+1]
        current_spread_history = spread_history_all[:i+1] if spread_history_all else []

        # Build DXY cumulative
        if dxy_aligned[i] is not None:
            dxy_cumulative.append(dxy_aligned[i])

        # Build implied vol cumulative
        if implied_vol_aligned[i] is not None:
            implied_vol_cumulative.append(implied_vol_aligned[i])

        # Build benchmark data
        benchmark = BenchmarkForexData(
            dxy_prices=dxy_cumulative.copy(),
            interest_rates=interest_rates,
            historical_rates=rates_df,
            cot_data=cot_df,
            calendar_events=calendar_events or [],
            implied_vol_prices=implied_vol_cumulative.copy(),
        )

        # Extract features
        features = extract_forex_features(
            row=df.iloc[i],
            symbol=symbol,
            benchmark_data=benchmark,
            pair_prices=current_prices,
            spread_history=current_spread_history,
            config=config,
        )

        # Update DataFrame
        df.iloc[i, df.columns.get_loc("carry_diff")] = features.rate_differential_norm if features.carry_valid else 0.0
        df.iloc[i, df.columns.get_loc("carry_regime")] = features.carry_regime if features.carry_valid else 0.0
        df.iloc[i, df.columns.get_loc("dxy_value")] = features.dxy_value if features.dxy_valid else 0.0
        df.iloc[i, df.columns.get_loc("dxy_rs")] = features.rs_vs_dxy_20d if features.dxy_valid else 0.0
        df.iloc[i, df.columns.get_loc("session_sydney")] = 1.0 if features.is_sydney else 0.0
        df.iloc[i, df.columns.get_loc("session_tokyo")] = 1.0 if features.is_tokyo else 0.0
        df.iloc[i, df.columns.get_loc("session_london")] = 1.0 if features.is_london else 0.0
        df.iloc[i, df.columns.get_loc("session_new_york")] = 1.0 if features.is_new_york else 0.0
        df.iloc[i, df.columns.get_loc("session_overlap")] = 1.0 if features.is_overlap else 0.0
        df.iloc[i, df.columns.get_loc("session_liquidity")] = features.session_liquidity if features.session_valid else 1.0
        df.iloc[i, df.columns.get_loc("spread_zscore")] = features.spread_zscore if features.spread_valid else 0.0
        df.iloc[i, df.columns.get_loc("spread_regime")] = features.spread_regime if features.spread_valid else 0.5
        df.iloc[i, df.columns.get_loc("cot_net")] = features.cot_net_long_pct if features.cot_valid else 0.5
        df.iloc[i, df.columns.get_loc("cot_zscore")] = features.cot_zscore if features.cot_valid else 0.0
        df.iloc[i, df.columns.get_loc("vol_5d")] = features.realized_vol_5d if features.vol_valid else 0.0
        df.iloc[i, df.columns.get_loc("vol_20d")] = features.realized_vol_20d if features.vol_valid else 0.0
        df.iloc[i, df.columns.get_loc("vol_ratio")] = features.vol_ratio if features.vol_valid else 1.0
        df.iloc[i, df.columns.get_loc("implied_vol")] = features.implied_vol if features.vol_valid else 0.0
        df.iloc[i, df.columns.get_loc("hours_to_event")] = min(features.hours_to_next_event, 999.0) if features.calendar_valid else 999.0
        df.iloc[i, df.columns.get_loc("is_news_window")] = 1.0 if features.is_news_window else 0.0

    return df


def _align_forex_benchmark(
    df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame],
    value_col: str = "close",
    timestamp_col: str = "timestamp",
) -> List[Optional[float]]:
    """
    Align benchmark data with main DataFrame using timestamp-based merge_asof.

    Same logic as stock_features._align_benchmark_by_timestamp() but for forex.
    """
    n_rows = len(df)

    if benchmark_df is None or benchmark_df.empty:
        return [None] * n_rows

    # Find timestamp column
    if timestamp_col not in df.columns:
        for alt_col in ["timestamp", "ts", "time", "date"]:
            if alt_col in df.columns:
                timestamp_col = alt_col
                break
        else:
            # Fall back to positional
            if value_col in benchmark_df.columns:
                vals = benchmark_df[value_col].tolist()
                return [vals[i] if i < len(vals) else None for i in range(n_rows)]
            return [None] * n_rows

    if timestamp_col not in benchmark_df.columns:
        if value_col in benchmark_df.columns:
            vals = benchmark_df[value_col].tolist()
            return [vals[i] if i < len(vals) else None for i in range(n_rows)]
        return [None] * n_rows

    # Find value column
    actual_value_col = None
    for col in [value_col, "close", "Close", "value", "mid"]:
        if col in benchmark_df.columns:
            actual_value_col = col
            break

    if actual_value_col is None:
        return [None] * n_rows

    # Prepare for merge_asof
    df_sorted = df[[timestamp_col]].copy()
    df_sorted = df_sorted.sort_values(timestamp_col).reset_index()
    df_sorted.rename(columns={"index": "_orig_idx"}, inplace=True)

    benchmark_sorted = benchmark_df[[timestamp_col, actual_value_col]].copy()
    benchmark_sorted = benchmark_sorted.dropna(subset=[timestamp_col, actual_value_col])
    benchmark_sorted = benchmark_sorted.sort_values(timestamp_col)
    benchmark_sorted.rename(columns={actual_value_col: "_benchmark_value"}, inplace=True)

    # merge_asof with backward direction
    merged = pd.merge_asof(
        df_sorted,
        benchmark_sorted,
        on=timestamp_col,
        direction="backward",
    )

    merged = merged.sort_values("_orig_idx")
    result = merged["_benchmark_value"].tolist()
    result = [None if pd.isna(v) else float(v) for v in result]

    return result
