# stock_features.py
"""
Stock-Specific Features Module for Phase 5: Features & Observation (Stock-specific)

This module provides stock-specific feature calculations that parallel the crypto
Fear & Greed index and other crypto-specific indicators.

Features implemented:
1. VIX Integration - Market volatility/fear indicator (analogous to Fear & Greed)
2. Market Regime Indicator - Bull/Bear/Sideways based on SPY/VIX
3. Sector/Industry Features - Sector rotation signals
4. Relative Strength - RS vs SPY/QQQ for momentum strategies

All features are designed for 100% backward compatibility with crypto:
- Crypto data without stock features will have validity=False
- Default values are sensible (0.0)
- No changes to existing crypto flow

Research references:
- CBOE VIX White Paper (2003): VIX as fear gauge
- Lo, A.W. (2004): "The Adaptive Markets Hypothesis"
- Moskowitz, T.J. et al. (2012): "Time series momentum" (relative strength)
- Levy, R. (1967): "Relative Strength as a Criterion for Investment Selection"

Author: Sultan Suyunbayev
Date: 2025-11-27
Version: 1.0.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONSTANTS
# =============================================================================

# VIX regime thresholds (based on historical analysis)
# References: CBOE VIX historical percentiles
VIX_REGIME_THRESHOLDS = {
    "low": 12.0,  # Below 12: Very calm market (complacency)
    "normal": 20.0,  # 12-20: Normal market conditions
    "elevated": 30.0,  # 20-30: Elevated fear
    "extreme": 40.0,  # Above 40: Extreme fear (crisis)
}

# Market regime thresholds
# Based on SMA crossover strategies and VIX interpretation
MARKET_REGIME_THRESHOLDS = {
    "bull_vix_max": 20.0,  # VIX must be below 20 for bull
    "bear_vix_min": 25.0,  # VIX must be above 25 for bear
    "trend_sma_fast": 20,  # 20-day SMA for fast trend
    "trend_sma_slow": 50,  # 50-day SMA for slow trend
}

# Relative strength lookback windows
RS_WINDOWS = {
    "short": 20,  # 20-day RS (approximately 1 month)
    "medium": 50,  # 50-day RS (approximately 2 months)
}

# Default values for missing data
DEFAULT_VIX = 20.0  # Historical VIX mean approximately 20
DEFAULT_MARKET_REGIME = 0.0  # Neutral/sideways
DEFAULT_RELATIVE_STRENGTH = 0.0  # Neutral


# =============================================================================
# ENUMS
# =============================================================================


class MarketRegime(Enum):
    """Market regime classification."""

    BEAR = -1
    SIDEWAYS = 0
    BULL = 1


class VIXRegime(Enum):
    """VIX-based volatility regime."""

    LOW = 0  # < 12: Complacency
    NORMAL = 1  # 12-20: Normal
    ELEVATED = 2  # 20-30: Elevated fear
    EXTREME = 3  # > 30: Extreme fear/crisis


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class StockFeatures:
    """
    Container for stock-specific features.

    All features include validity flags to distinguish missing data from zero values,
    following the pattern established by Fear & Greed in crypto.
    """

    # VIX features
    vix_value: float = DEFAULT_VIX
    vix_valid: bool = False
    vix_regime: float = 0.5  # Normalized regime (0-1)
    vix_regime_valid: bool = False

    # Market regime
    market_regime: float = DEFAULT_MARKET_REGIME  # -1 to 1
    market_regime_valid: bool = False

    # Relative strength vs benchmarks
    rs_spy_20d: float = DEFAULT_RELATIVE_STRENGTH
    rs_spy_20d_valid: bool = False
    rs_spy_50d: float = DEFAULT_RELATIVE_STRENGTH
    rs_spy_50d_valid: bool = False
    rs_qqq_20d: float = DEFAULT_RELATIVE_STRENGTH
    rs_qqq_20d_valid: bool = False

    # Sector features
    sector_momentum: float = 0.0
    sector_momentum_valid: bool = False


@dataclass
class BenchmarkData:
    """
    Container for benchmark price data (SPY, QQQ, VIX).

    This is used to pass benchmark data to feature calculators.
    """

    spy_prices: List[float] = field(default_factory=list)
    qqq_prices: List[float] = field(default_factory=list)
    vix_values: List[float] = field(default_factory=list)
    sector_returns: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# VIX FEATURES
# =============================================================================


def calculate_vix_regime(vix_value: float) -> Tuple[float, VIXRegime]:
    """
    Calculate VIX regime from VIX value.

    The VIX Index is a measure of expected market volatility over the next 30 days.
    Different VIX levels correspond to different market conditions:

    - LOW (< 12): Market complacency, often seen before corrections
    - NORMAL (12-20): Typical market conditions
    - ELEVATED (20-30): Heightened uncertainty
    - EXTREME (> 30): Significant fear, often crisis periods

    Args:
        vix_value: Current VIX value

    Returns:
        Tuple of (normalized_regime, VIXRegime enum)
        - normalized_regime: 0.0 (low fear) to 1.0 (extreme fear)
        - VIXRegime: categorical classification

    Research references:
    - Whaley, R.E. (2000): "The Investor Fear Gauge"
    - CBOE VIX White Paper (2003)
    """
    if not math.isfinite(vix_value) or vix_value < 0:
        return 0.5, VIXRegime.NORMAL

    # Determine regime
    if vix_value < VIX_REGIME_THRESHOLDS["low"]:
        regime = VIXRegime.LOW
        # Normalize: 0-12 maps to 0.0-0.25
        normalized = vix_value / VIX_REGIME_THRESHOLDS["low"] * 0.25
    elif vix_value < VIX_REGIME_THRESHOLDS["normal"]:
        regime = VIXRegime.NORMAL
        # Normalize: 12-20 maps to 0.25-0.5
        range_val = vix_value - VIX_REGIME_THRESHOLDS["low"]
        range_size = VIX_REGIME_THRESHOLDS["normal"] - VIX_REGIME_THRESHOLDS["low"]
        normalized = 0.25 + (range_val / range_size) * 0.25
    elif vix_value < VIX_REGIME_THRESHOLDS["elevated"]:
        regime = VIXRegime.ELEVATED
        # Normalize: 20-30 maps to 0.5-0.75
        range_val = vix_value - VIX_REGIME_THRESHOLDS["normal"]
        range_size = VIX_REGIME_THRESHOLDS["elevated"] - VIX_REGIME_THRESHOLDS["normal"]
        normalized = 0.5 + (range_val / range_size) * 0.25
    else:
        regime = VIXRegime.EXTREME
        # Normalize: 30+ maps to 0.75-1.0 (capped at VIX=80)
        range_val = min(vix_value - VIX_REGIME_THRESHOLDS["elevated"], 50.0)
        range_size = 50.0  # Cap at VIX=80
        normalized = 0.75 + (range_val / range_size) * 0.25

    return min(max(normalized, 0.0), 1.0), regime


def normalize_vix_value(vix_value: float) -> float:
    """
    Normalize VIX value for observation vector.

    Uses tanh transformation to bound the value similar to other features.
    Historical VIX range: approximately 9 (minimum) to 80+ (crisis maximum).

    Normalization: tanh((vix - 20) / 20)
    - VIX = 0:  tanh(-1) ≈ -0.76
    - VIX = 10: tanh(-0.5) ≈ -0.46
    - VIX = 20: tanh(0) = 0
    - VIX = 40: tanh(1) ≈ 0.76
    - VIX = 60: tanh(2) ≈ 0.96

    Args:
        vix_value: Raw VIX value

    Returns:
        Normalized VIX in approximately [-1, 1] range
    """
    if not math.isfinite(vix_value):
        return 0.0

    # Center at typical VIX (20), scale by 20
    normalized = math.tanh((vix_value - 20.0) / 20.0)
    return normalized


# =============================================================================
# MARKET REGIME FEATURES
# =============================================================================


def calculate_market_regime(
    spy_prices: List[float],
    vix_value: Optional[float] = None,
    sma_fast_window: int = 20,
    sma_slow_window: int = 50,
) -> Tuple[float, MarketRegime, bool]:
    """
    Calculate market regime based on SPY trend and VIX level.

    Market regime is determined by:
    1. SPY SMA crossover: Fast SMA (20) vs Slow SMA (50)
    2. VIX level: High VIX overrides to BEAR regime

    Logic:
    - BULL: Fast SMA > Slow SMA AND VIX < 20
    - BEAR: Fast SMA < Slow SMA AND VIX > 25, OR VIX > 35
    - SIDEWAYS: Everything else

    Args:
        spy_prices: List of SPY close prices (most recent last)
        vix_value: Current VIX value (optional)
        sma_fast_window: Fast SMA window (default 20)
        sma_slow_window: Slow SMA window (default 50)

    Returns:
        Tuple of (regime_value, MarketRegime, is_valid)
        - regime_value: -1.0 (bear) to 1.0 (bull)
        - MarketRegime: categorical classification
        - is_valid: Whether calculation was successful

    Research references:
    - Golden Cross/Death Cross strategy
    - VIX as market timing signal (Karpoff, 2008)
    """
    # Check if we have enough data
    if not spy_prices or len(spy_prices) < sma_slow_window:
        return DEFAULT_MARKET_REGIME, MarketRegime.SIDEWAYS, False

    try:
        # Calculate SMAs
        prices_arr = np.array(spy_prices[-sma_slow_window:], dtype=np.float64)

        if len(prices_arr) < sma_slow_window or not np.all(np.isfinite(prices_arr)):
            return DEFAULT_MARKET_REGIME, MarketRegime.SIDEWAYS, False

        sma_slow = np.mean(prices_arr)
        sma_fast = np.mean(prices_arr[-sma_fast_window:])

        if not (math.isfinite(sma_slow) and math.isfinite(sma_fast)):
            return DEFAULT_MARKET_REGIME, MarketRegime.SIDEWAYS, False

        # Calculate trend strength (percentage difference)
        trend_strength = (sma_fast - sma_slow) / (sma_slow + 1e-8) * 100

        # VIX override: extreme VIX indicates crisis regardless of trend
        vix_override_bear = False
        vix_override_neutral = False

        if vix_value is not None and math.isfinite(vix_value):
            if vix_value > 35:
                vix_override_bear = True
            elif vix_value > MARKET_REGIME_THRESHOLDS["bear_vix_min"]:
                vix_override_neutral = True

        # Determine regime
        if vix_override_bear:
            regime = MarketRegime.BEAR
            regime_value = -1.0
        elif vix_override_neutral and trend_strength < 0:
            regime = MarketRegime.BEAR
            # Scale to [-1, -0.5] based on trend strength
            regime_value = max(-1.0, -0.5 + min(0, trend_strength / 10))
        elif trend_strength > 1.0:  # Bullish trend (SMA fast > slow by 1%+)
            vix_ok = vix_value is None or vix_value < MARKET_REGIME_THRESHOLDS["bull_vix_max"]
            if vix_ok:
                regime = MarketRegime.BULL
                # Scale to [0.5, 1.0] based on trend strength
                regime_value = min(1.0, 0.5 + min(trend_strength / 10, 0.5))
            else:
                regime = MarketRegime.SIDEWAYS
                regime_value = min(0.5, trend_strength / 10)
        elif trend_strength < -1.0:  # Bearish trend (SMA fast < slow by 1%+)
            regime = MarketRegime.BEAR
            # Scale to [-1.0, -0.5] based on trend strength
            regime_value = max(-1.0, -0.5 + max(trend_strength / 10, -0.5))
        else:
            regime = MarketRegime.SIDEWAYS
            regime_value = trend_strength / 5  # Slight lean based on trend
            regime_value = max(-0.5, min(0.5, regime_value))

        return regime_value, regime, True

    except Exception:
        return DEFAULT_MARKET_REGIME, MarketRegime.SIDEWAYS, False


# =============================================================================
# RELATIVE STRENGTH FEATURES
# =============================================================================


def calculate_relative_strength(
    stock_prices: List[float],
    benchmark_prices: List[float],
    window: int = 20,
) -> Tuple[float, bool]:
    """
    Calculate relative strength of a stock vs benchmark.

    Relative strength measures a stock's performance relative to a benchmark
    over a specified period. Positive RS indicates outperformance.

    Formula:
    RS = (stock_return / benchmark_return) - 1

    Where returns are calculated as: (price_now / price_n_days_ago) - 1

    Normalization:
    - Raw RS can be any value (e.g., stock +10%, benchmark +5% → RS = 4.76%)
    - We normalize using tanh to bound to [-1, 1]
    - Scale factor of 5 means ±20% RS saturates to ±1

    Args:
        stock_prices: Stock close prices (most recent last)
        benchmark_prices: Benchmark close prices (most recent last)
        window: Lookback period for RS calculation

    Returns:
        Tuple of (rs_value, is_valid)
        - rs_value: Normalized RS in approximately [-1, 1] range
        - is_valid: Whether calculation was successful

    Research references:
    - Levy, R. (1967): "Relative Strength as a Criterion for Investment Selection"
    - Jegadeesh & Titman (1993): "Returns to Buying Winners and Selling Losers"
    - Moskowitz et al. (2012): "Time series momentum"
    """
    if (
        not stock_prices
        or not benchmark_prices
        or len(stock_prices) < window + 1
        or len(benchmark_prices) < window + 1
    ):
        return DEFAULT_RELATIVE_STRENGTH, False

    try:
        # Get prices at both ends of the window
        stock_now = float(stock_prices[-1])
        stock_past = float(stock_prices[-(window + 1)])
        benchmark_now = float(benchmark_prices[-1])
        benchmark_past = float(benchmark_prices[-(window + 1)])

        # Validate prices
        if not all(
            math.isfinite(p) and p > 0
            for p in [stock_now, stock_past, benchmark_now, benchmark_past]
        ):
            return DEFAULT_RELATIVE_STRENGTH, False

        # Calculate returns
        stock_return = (stock_now / stock_past) - 1.0
        benchmark_return = (benchmark_now / benchmark_past) - 1.0

        # Handle edge case: benchmark return near zero
        if abs(benchmark_return) < 1e-8:
            # Benchmark flat, use stock return directly
            raw_rs = stock_return
        else:
            # RS = (1 + stock_ret) / (1 + benchmark_ret) - 1
            # This handles negative returns correctly
            raw_rs = (1.0 + stock_return) / (1.0 + benchmark_return) - 1.0

        if not math.isfinite(raw_rs):
            return DEFAULT_RELATIVE_STRENGTH, False

        # Normalize using tanh with scale factor
        # ±20% RS → ±0.96 (near saturation)
        normalized_rs = math.tanh(raw_rs * 5.0)

        return normalized_rs, True

    except Exception:
        return DEFAULT_RELATIVE_STRENGTH, False


# =============================================================================
# SECTOR FEATURES
# =============================================================================

# Sector ETF mapping (standard GICS sectors)
SECTOR_ETFS = {
    "technology": "XLK",
    "healthcare": "XLV",
    "financials": "XLF",
    "consumer_discretionary": "XLY",
    "consumer_staples": "XLP",
    "industrials": "XLI",
    "materials": "XLB",
    "energy": "XLE",
    "utilities": "XLU",
    "real_estate": "XLRE",
    "communication_services": "XLC",
}

# Symbol to sector mapping (common stocks)
# This would typically come from exchange_info adapter
SYMBOL_TO_SECTOR = {
    # Technology
    "AAPL": "technology",
    "MSFT": "technology",
    "GOOGL": "technology",
    "GOOG": "technology",
    "META": "technology",
    "NVDA": "technology",
    "AMD": "technology",
    "INTC": "technology",
    "CRM": "technology",
    "ADBE": "technology",
    "ORCL": "technology",
    "CSCO": "technology",
    # Healthcare
    "JNJ": "healthcare",
    "UNH": "healthcare",
    "PFE": "healthcare",
    "MRK": "healthcare",
    "ABBV": "healthcare",
    "LLY": "healthcare",
    # Financials
    "JPM": "financials",
    "BAC": "financials",
    "WFC": "financials",
    "GS": "financials",
    "MS": "financials",
    "C": "financials",
    # Consumer Discretionary
    "AMZN": "consumer_discretionary",
    "TSLA": "consumer_discretionary",
    "HD": "consumer_discretionary",
    "NKE": "consumer_discretionary",
    # Consumer Staples
    "PG": "consumer_staples",
    "KO": "consumer_staples",
    "PEP": "consumer_staples",
    "WMT": "consumer_staples",
    "COST": "consumer_staples",
    # Industrials
    "BA": "industrials",
    "CAT": "industrials",
    "HON": "industrials",
    "UPS": "industrials",
    "GE": "industrials",
    # Energy
    "XOM": "energy",
    "CVX": "energy",
    "COP": "energy",
    # Communication Services
    "DIS": "communication_services",
    "NFLX": "communication_services",
    "T": "communication_services",
    "VZ": "communication_services",
}


def calculate_sector_momentum(
    symbol: str,
    sector_returns: Dict[str, float],
    market_return: float = 0.0,
) -> Tuple[float, bool]:
    """
    Calculate sector momentum relative to market.

    Sector momentum measures how a stock's sector is performing relative
    to the overall market. Stocks in outperforming sectors tend to continue
    outperforming (sector rotation effect).

    Args:
        symbol: Stock ticker symbol
        sector_returns: Dict mapping sector names to their returns
        market_return: Overall market return (e.g., SPY return)

    Returns:
        Tuple of (sector_momentum, is_valid)
        - sector_momentum: Normalized sector excess return in [-1, 1]
        - is_valid: Whether calculation was successful

    Research references:
    - Moskowitz & Grinblatt (1999): "Do Industries Explain Momentum?"
    - Menzly & Ozbas (2010): "Market Segmentation and Cross-predictability"
    """
    # Get sector for this symbol
    sector = get_symbol_sector(symbol)
    if sector is None or not sector_returns:
        return 0.0, False

    try:
        # Get sector return
        sector_return = sector_returns.get(sector)
        if sector_return is None or not math.isfinite(sector_return):
            return 0.0, False

        # Calculate excess return vs market
        if not math.isfinite(market_return):
            market_return = 0.0

        excess_return = sector_return - market_return

        # Normalize using tanh
        # ±5% excess return → ±0.96
        normalized = math.tanh(excess_return * 10.0)

        return normalized, True

    except Exception:
        return 0.0, False


def get_symbol_sector(symbol: str) -> Optional[str]:
    """
    Get sector classification for a symbol.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Sector name or None if unknown
    """
    return SYMBOL_TO_SECTOR.get(symbol.upper())


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================


def extract_stock_features(
    row: Any,
    symbol: Optional[str] = None,
    benchmark_data: Optional[BenchmarkData] = None,
    stock_prices: Optional[List[float]] = None,
) -> StockFeatures:
    """
    Extract all stock-specific features from a data row.

    This is the main entry point for stock feature extraction.
    Features are designed to be analogous to crypto's Fear & Greed index.

    Args:
        row: Data row (DataFrame row or dict) with market data
        symbol: Stock ticker symbol (for sector classification)
        benchmark_data: Container with SPY, QQQ, VIX data
        stock_prices: Historical prices for the stock (for RS calculation)

    Returns:
        StockFeatures dataclass with all extracted features and validity flags

    Example:
        >>> benchmark = BenchmarkData(
        ...     spy_prices=spy_close_history,
        ...     qqq_prices=qqq_close_history,
        ...     vix_values=vix_history,
        ... )
        >>> features = extract_stock_features(row, "AAPL", benchmark, aapl_prices)
        >>> print(f"VIX: {features.vix_value}, Regime: {features.market_regime}")
    """
    features = StockFeatures()

    # Extract VIX from row or benchmark data
    vix_value = _extract_vix_from_row(row)
    if vix_value is None and benchmark_data and benchmark_data.vix_values:
        vix_value = benchmark_data.vix_values[-1] if benchmark_data.vix_values else None

    if vix_value is not None and math.isfinite(vix_value):
        features.vix_value = normalize_vix_value(vix_value)
        features.vix_valid = True

        vix_regime_norm, _ = calculate_vix_regime(vix_value)
        features.vix_regime = vix_regime_norm
        features.vix_regime_valid = True

    # Calculate market regime
    if benchmark_data and benchmark_data.spy_prices:
        regime_val, _, regime_valid = calculate_market_regime(
            benchmark_data.spy_prices,
            vix_value if features.vix_valid else None,
        )
        if regime_valid:
            features.market_regime = regime_val
            features.market_regime_valid = True

    # Calculate relative strength vs SPY
    if stock_prices and benchmark_data and benchmark_data.spy_prices:
        rs_20d, valid_20d = calculate_relative_strength(
            stock_prices, benchmark_data.spy_prices, window=20
        )
        if valid_20d:
            features.rs_spy_20d = rs_20d
            features.rs_spy_20d_valid = True

        rs_50d, valid_50d = calculate_relative_strength(
            stock_prices, benchmark_data.spy_prices, window=50
        )
        if valid_50d:
            features.rs_spy_50d = rs_50d
            features.rs_spy_50d_valid = True

    # Calculate relative strength vs QQQ
    if stock_prices and benchmark_data and benchmark_data.qqq_prices:
        rs_qqq_20d, valid_qqq = calculate_relative_strength(
            stock_prices, benchmark_data.qqq_prices, window=20
        )
        if valid_qqq:
            features.rs_qqq_20d = rs_qqq_20d
            features.rs_qqq_20d_valid = True

    # Calculate sector momentum
    if symbol and benchmark_data and benchmark_data.sector_returns:
        # Get market return (SPY return approximation)
        market_return = 0.0
        if benchmark_data.spy_prices and len(benchmark_data.spy_prices) >= 2:
            market_return = benchmark_data.spy_prices[-1] / benchmark_data.spy_prices[-2] - 1

        sector_mom, sector_valid = calculate_sector_momentum(
            symbol, benchmark_data.sector_returns, market_return
        )
        if sector_valid:
            features.sector_momentum = sector_mom
            features.sector_momentum_valid = True

    return features


def _extract_vix_from_row(row: Any) -> Optional[float]:
    """Extract VIX value from data row."""
    if row is None:
        return None

    # Try different column names
    vix_columns = ["vix", "vix_close", "VIX", "vix_value", "^VIX"]

    for col in vix_columns:
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

    return None


# =============================================================================
# DATAFRAME UTILITIES
# =============================================================================


def _align_benchmark_by_timestamp(
    df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame],
    value_col: str = "close",
    timestamp_col: str = "timestamp",
) -> List[Optional[float]]:
    """
    Align benchmark data with main DataFrame using timestamp-based merge_asof.

    FIX (2025-11-29): This function replaces positional indexing with proper
    temporal alignment to prevent look-ahead bias when benchmark data has
    different date ranges or gaps than the main data.

    The issue was that the previous implementation used positional indexing:
        benchmark_values[i] for row i
    This fails when:
    - Benchmark data starts earlier/later than main data
    - Benchmark has different gaps
    - Timestamps don't align exactly

    The fix uses merge_asof with direction="backward" which:
    - Finds the nearest benchmark value <= each timestamp
    - Ensures only PAST information is used (no look-ahead)
    - Handles misaligned timestamps correctly

    Args:
        df: Main DataFrame with timestamp column
        benchmark_df: Benchmark DataFrame to align (SPY, QQQ, VIX)
        value_col: Column to extract values from
        timestamp_col: Timestamp column name

    Returns:
        List of aligned values (same length as df), None for missing

    Reference: López de Prado (2018) Ch.4 - Proper temporal alignment in financial ML
    """
    n_rows = len(df)

    if benchmark_df is None or benchmark_df.empty:
        return [None] * n_rows

    # Ensure timestamp column exists
    if timestamp_col not in df.columns:
        # Try alternative timestamp columns
        for alt_col in ["timestamp", "ts", "time", "date"]:
            if alt_col in df.columns:
                timestamp_col = alt_col
                break
        else:
            # No timestamp column - fall back to positional (with warning)
            import warnings

            warnings.warn(
                "No timestamp column found in DataFrame. "
                "Falling back to positional indexing which may cause temporal misalignment. "
                "Add 'timestamp' column for proper alignment.",
                RuntimeWarning,
                stacklevel=2,
            )
            if value_col in benchmark_df.columns:
                vals = benchmark_df[value_col].tolist()
                return [vals[i] if i < len(vals) else None for i in range(n_rows)]
            return [None] * n_rows

    if timestamp_col not in benchmark_df.columns:
        # Benchmark has no timestamp - fall back to positional
        if value_col in benchmark_df.columns:
            vals = benchmark_df[value_col].tolist()
            return [vals[i] if i < len(vals) else None for i in range(n_rows)]
        return [None] * n_rows

    # Find value column in benchmark
    actual_value_col = None
    for col_candidate in [value_col, "close", "Close", "VIX", "vix_close", "value"]:
        if col_candidate in benchmark_df.columns:
            actual_value_col = col_candidate
            break

    if actual_value_col is None:
        return [None] * n_rows

    # Prepare data for merge_asof
    df_sorted = df[[timestamp_col]].copy()
    df_sorted = df_sorted.sort_values(timestamp_col).reset_index()
    df_sorted.rename(columns={"index": "_orig_idx"}, inplace=True)

    benchmark_sorted = benchmark_df[[timestamp_col, actual_value_col]].copy()
    benchmark_sorted = benchmark_sorted.dropna(subset=[timestamp_col, actual_value_col])
    benchmark_sorted = benchmark_sorted.sort_values(timestamp_col)
    benchmark_sorted.rename(columns={actual_value_col: "_benchmark_value"}, inplace=True)

    # Perform merge_asof with direction="backward"
    # This ensures we only use benchmark values from BEFORE or AT the timestamp
    merged = pd.merge_asof(
        df_sorted,
        benchmark_sorted,
        on=timestamp_col,
        direction="backward",
    )

    # Restore original order
    merged = merged.sort_values("_orig_idx")

    # Extract values in original order
    result = merged["_benchmark_value"].tolist()

    # Convert NaN to None for consistency
    result = [None if pd.isna(v) else float(v) for v in result]

    return result


def add_stock_features_to_dataframe(
    df: pd.DataFrame,
    symbol: str,
    spy_df: Optional[pd.DataFrame] = None,
    qqq_df: Optional[pd.DataFrame] = None,
    vix_df: Optional[pd.DataFrame] = None,
    sector_returns_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Add stock-specific features to a DataFrame.

    This function adds all stock features as new columns to the input DataFrame.
    Used during data preparation for training/inference.

    FIX (2025-11-29): Now uses timestamp-based merge_asof for benchmark alignment
    instead of positional indexing. This prevents temporal misalignment when
    benchmark data (VIX/SPY/QQQ) has different date ranges than the main data.

    Args:
        df: Main DataFrame with OHLCV data
        symbol: Stock ticker symbol
        spy_df: SPY price DataFrame (optional, for RS and regime)
        qqq_df: QQQ price DataFrame (optional, for RS)
        vix_df: VIX DataFrame (optional, for VIX features)
        sector_returns_df: Sector returns DataFrame (optional)

    Returns:
        DataFrame with added stock feature columns

    New columns added:
    - vix_normalized: Normalized VIX value
    - vix_regime: VIX regime (0-1 scale)
    - market_regime: Bull/Bear/Sideways (-1 to 1)
    - rs_spy_20d: 20-day relative strength vs SPY
    - rs_spy_50d: 50-day relative strength vs SPY
    - rs_qqq_20d: 20-day relative strength vs QQQ
    - sector_momentum: Sector momentum relative to market
    """
    df = df.copy()
    n_rows = len(df)

    # Initialize columns with default values
    df["vix_normalized"] = 0.0
    df["vix_regime"] = 0.5
    df["market_regime"] = 0.0
    df["rs_spy_20d"] = 0.0
    df["rs_spy_50d"] = 0.0
    df["rs_qqq_20d"] = 0.0
    df["sector_momentum"] = 0.0

    # Get close prices
    close_col = "close" if "close" in df.columns else "Close"
    if close_col not in df.columns:
        return df

    stock_prices = df[close_col].tolist()

    # FIX (2025-11-29): Use timestamp-based alignment instead of positional indexing
    # This prevents temporal misalignment when benchmark data has different date ranges
    #
    # The old code used positional indexing:
    #   spy_prices = spy_df["close"].tolist()
    #   current_spy_prices = spy_prices[:i+1]  # BUG: position i != timestamp at row i
    #
    # The fix aligns benchmark data by timestamp FIRST, then uses the aligned values:
    #   spy_aligned = _align_benchmark_by_timestamp(df, spy_df, "close")
    #   current_spy = spy_aligned[i]  # CORRECT: aligned by timestamp
    spy_aligned = _align_benchmark_by_timestamp(df, spy_df, "close")
    qqq_aligned = _align_benchmark_by_timestamp(df, qqq_df, "close")
    vix_aligned = _align_benchmark_by_timestamp(df, vix_df, "close")

    # Build cumulative lists from aligned values for rolling calculations
    # For relative strength, we need historical prices aligned by timestamp
    spy_cumulative: List[float] = []
    qqq_cumulative: List[float] = []

    # Calculate rolling features
    for i in range(n_rows):
        # Build historical data up to current point using ALIGNED values
        current_stock_prices = stock_prices[: i + 1]

        # FIX: Build cumulative benchmark prices from aligned values
        if spy_aligned[i] is not None:
            spy_cumulative.append(spy_aligned[i])
        current_spy_prices = spy_cumulative.copy()

        if qqq_aligned[i] is not None:
            qqq_cumulative.append(qqq_aligned[i])
        current_qqq_prices = qqq_cumulative.copy()

        # FIX: Use aligned VIX value at current position
        current_vix = vix_aligned[i]

        # Build benchmark data
        benchmark = BenchmarkData(
            spy_prices=current_spy_prices,
            qqq_prices=current_qqq_prices,
            vix_values=[current_vix] if current_vix is not None else [],
        )

        # Extract features
        features = extract_stock_features(
            row=df.iloc[i],
            symbol=symbol,
            benchmark_data=benchmark,
            stock_prices=current_stock_prices,
        )

        # Update DataFrame
        df.iloc[i, df.columns.get_loc("vix_normalized")] = (
            features.vix_value if features.vix_valid else 0.0
        )
        df.iloc[i, df.columns.get_loc("vix_regime")] = (
            features.vix_regime if features.vix_regime_valid else 0.5
        )
        df.iloc[i, df.columns.get_loc("market_regime")] = (
            features.market_regime if features.market_regime_valid else 0.0
        )
        df.iloc[i, df.columns.get_loc("rs_spy_20d")] = (
            features.rs_spy_20d if features.rs_spy_20d_valid else 0.0
        )
        df.iloc[i, df.columns.get_loc("rs_spy_50d")] = (
            features.rs_spy_50d if features.rs_spy_50d_valid else 0.0
        )
        df.iloc[i, df.columns.get_loc("rs_qqq_20d")] = (
            features.rs_qqq_20d if features.rs_qqq_20d_valid else 0.0
        )
        df.iloc[i, df.columns.get_loc("sector_momentum")] = (
            features.sector_momentum if features.sector_momentum_valid else 0.0
        )

    return df


def calculate_sector_returns_from_etfs(
    etf_data: Dict[str, pd.DataFrame],
    window: int = 20,
) -> Dict[str, float]:
    """
    Calculate sector returns from sector ETF data.

    Args:
        etf_data: Dict mapping sector ETF symbol to price DataFrame
        window: Return calculation window

    Returns:
        Dict mapping sector name to return value
    """
    sector_returns = {}

    # Reverse mapping: ETF symbol -> sector name
    etf_to_sector = {v: k for k, v in SECTOR_ETFS.items()}

    for etf_symbol, df in etf_data.items():
        sector = etf_to_sector.get(etf_symbol)
        if sector is None:
            continue

        close_col = "close" if "close" in df.columns else "Close"
        if close_col not in df.columns or len(df) < window + 1:
            continue

        try:
            price_now = float(df[close_col].iloc[-1])
            price_past = float(df[close_col].iloc[-(window + 1)])

            if price_past > 0 and math.isfinite(price_now) and math.isfinite(price_past):
                sector_returns[sector] = (price_now / price_past) - 1.0
        except Exception:
            continue

    return sector_returns
