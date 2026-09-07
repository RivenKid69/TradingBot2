# -*- coding: utf-8 -*-
"""
services/trading_halts.py
Trading halts simulation for US Equities.

This module implements:
1. LULD (Limit Up Limit Down) circuit breakers - price band violations
2. Market-wide circuit breakers - S&P 500 triggers (Level 1/2/3)
3. News halts - pending announcement halts (simulated)
4. Regulatory suspensions - SEC/exchange halts

References:
- SEC Rule 201 (Alternative Uptick Rule): https://www.sec.gov/rules/final/2010/34-61595.pdf
- NYSE LULD FAQ: https://www.nyse.com/publicdocs/nyse/markets/nyse/LULD_FAQ.pdf
- Market-Wide Circuit Breakers: https://www.nyse.com/markets/nyse/trading-info/circuit-breakers

LULD Bands (Tier 1 - S&P 500, Russell 1000):
- 9:30-9:45 ET: 10% bands
- 9:45-15:35 ET: 5% bands
- 15:35-16:00 ET: 10% bands

LULD Bands (Tier 2 - all others):
- All day: 10% bands (15 second pause)

Market-Wide Circuit Breakers (S&P 500 decline from previous close):
- Level 1: 7% decline - 15 min halt (only before 3:25pm ET)
- Level 2: 13% decline - 15 min halt (only before 3:25pm ET)
- Level 3: 20% decline - halt for remainder of day

Design Principles:
- All halts are asset-class aware (skip for crypto)
- Backward compatible with existing ExecutionSimulator
- Supports both real-time and backtesting modes
- Thread-safe for multi-symbol trading
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, time, timezone, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Set

logger = logging.getLogger(__name__)


# =========================
# Constants
# =========================

# LULD Band percentages
LULD_TIER1_OPEN_BAND = 0.10  # 10% during open/close
LULD_TIER1_REGULAR_BAND = 0.05  # 5% during regular hours
LULD_TIER2_BAND = 0.10  # 10% all day

# Market-wide circuit breaker thresholds
MWCB_LEVEL1_THRESHOLD = -0.07  # 7% decline
MWCB_LEVEL2_THRESHOLD = -0.13  # 13% decline
MWCB_LEVEL3_THRESHOLD = -0.20  # 20% decline

# Halt durations
LULD_HALT_DURATION_SEC = 15 * 60  # 15 minutes for LULD pause
MWCB_HALT_DURATION_SEC = 15 * 60  # 15 minutes for MWCB Level 1/2

# NYSE session times (Eastern Time)
NYSE_OPEN = time(9, 30)
NYSE_EARLY_OPEN_END = time(9, 45)
NYSE_LATE_START = time(15, 35)
NYSE_CLOSE = time(16, 0)
NYSE_MWCB_CUTOFF = time(15, 25)


# =========================
# Enumerations
# =========================


class HaltType(str, Enum):
    """Types of trading halts."""

    NONE = "NONE"
    LULD_PAUSE = "LULD_PAUSE"  # Limit Up/Limit Down pause
    LULD_HALT = "LULD_HALT"  # Extended LULD halt
    MWCB_LEVEL1 = "MWCB_LEVEL1"  # Market-wide 7% decline
    MWCB_LEVEL2 = "MWCB_LEVEL2"  # Market-wide 13% decline
    MWCB_LEVEL3 = "MWCB_LEVEL3"  # Market-wide 20% decline
    NEWS_PENDING = "NEWS_PENDING"  # Pending news announcement
    REGULATORY = "REGULATORY"  # SEC/Exchange suspension
    VOLATILITY = "VOLATILITY"  # Exchange volatility pause


class TierType(str, Enum):
    """LULD tier classification."""

    TIER1 = "TIER1"  # S&P 500, Russell 1000, ETFs
    TIER2 = "TIER2"  # All other NMS securities


# =========================
# Data Classes
# =========================


@dataclass
class LULDBands:
    """LULD price bands for a symbol."""

    symbol: str
    reference_price: float  # Average of last 5 minutes
    upper_band: float  # Limit up price
    lower_band: float  # Limit down price
    band_percentage: float  # Current band percentage
    tier: TierType = TierType.TIER2
    last_update_ms: int = 0


@dataclass
class HaltStatus:
    """Current halt status for a symbol or market."""

    symbol: str  # "MARKET" for market-wide
    halt_type: HaltType = HaltType.NONE
    is_halted: bool = False
    halt_start_ms: Optional[int] = None
    halt_end_ms: Optional[int] = None
    reason: str = ""
    trigger_price: Optional[float] = None
    resume_price: Optional[float] = None


@dataclass
class TradingHaltsConfig:
    """Configuration for TradingHaltsSimulator."""

    # LULD settings
    luld_enabled: bool = True
    luld_pause_duration_sec: int = LULD_HALT_DURATION_SEC

    # Market-wide circuit breakers
    mwcb_enabled: bool = True
    mwcb_halt_duration_sec: int = MWCB_HALT_DURATION_SEC

    # News halts (simulated based on volatility)
    news_halts_enabled: bool = False  # Disabled by default for backtesting
    news_halt_volatility_threshold: float = 0.10  # 10% intraday move

    # General
    enabled: bool = True
    asset_class: str = "equity"  # Only applies to equity

    # Tier 1 symbols (simplified - in production, load from NYSE list)
    tier1_symbols: Set[str] = field(
        default_factory=lambda: {
            "SPY",
            "QQQ",
            "IWM",
            "DIA",  # Major ETFs
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "NVDA",
            "TSLA",  # Mega caps
            "GLD",
            "SLV",  # Commodity ETFs
        }
    )


# =========================
# Trading Halts Simulator
# =========================


class TradingHaltsSimulator:
    """
    Simulates trading halts for US equities.

    Implements LULD price bands, market-wide circuit breakers,
    and other regulatory halt mechanisms.

    Usage:
        halts = TradingHaltsSimulator(config)

        # Check if trading is allowed
        status = halts.check_halt_status("AAPL", price=150.0, timestamp_ms=ts)
        if status.is_halted:
            # Skip trade or queue for later
            pass

        # Update reference price (every 5 minutes)
        halts.update_reference_price("AAPL", avg_price=150.5, timestamp_ms=ts)

        # Check market-wide status
        market_status = halts.check_market_wide_halt(spx_return=-0.08)
    """

    def __init__(
        self,
        config: Optional[TradingHaltsConfig] = None,
    ) -> None:
        """
        Initialize TradingHaltsSimulator.

        Args:
            config: Configuration for halt simulation
        """
        self._config = config or TradingHaltsConfig()

        # Symbol-level state
        self._luld_bands: Dict[str, LULDBands] = {}
        self._symbol_halts: Dict[str, HaltStatus] = {}
        self._reference_prices: Dict[str, float] = {}
        self._prev_close_prices: Dict[str, float] = {}

        # Market-wide state
        self._market_halt: HaltStatus = HaltStatus(symbol="MARKET")
        self._mwcb_triggered_today: Set[str] = set()  # Track which levels triggered
        self._trading_day_start_ms: int = 0

        # Thread safety
        self._lock = threading.RLock()

        logger.debug(
            f"TradingHaltsSimulator initialized: luld={self._config.luld_enabled}, "
            f"mwcb={self._config.mwcb_enabled}"
        )

    # =========================
    # Configuration
    # =========================

    def set_prev_close(self, symbol: str, price: float) -> None:
        """Set previous day's closing price for a symbol."""
        with self._lock:
            self._prev_close_prices[symbol.upper()] = float(price)

    def set_reference_price(self, symbol: str, price: float, timestamp_ms: int) -> None:
        """Set LULD reference price (typically 5-minute avg)."""
        with self._lock:
            symbol = symbol.upper()
            self._reference_prices[symbol] = float(price)
            self._update_luld_bands(symbol, price, timestamp_ms)

    def set_tier1_symbols(self, symbols: Set[str]) -> None:
        """Set Tier 1 symbols (S&P 500, Russell 1000, etc.)."""
        with self._lock:
            self._config.tier1_symbols = {s.upper() for s in symbols}

    def reset_daily_state(self, trading_day_start_ms: int) -> None:
        """Reset daily state at market open."""
        with self._lock:
            self._mwcb_triggered_today.clear()
            self._trading_day_start_ms = trading_day_start_ms
            self._market_halt = HaltStatus(symbol="MARKET")

            # Clear symbol halts
            self._symbol_halts.clear()
            self._luld_bands.clear()

    # =========================
    # LULD Price Bands
    # =========================

    def _get_tier(self, symbol: str) -> TierType:
        """Get LULD tier for a symbol."""
        return TierType.TIER1 if symbol.upper() in self._config.tier1_symbols else TierType.TIER2

    def _get_luld_band_percentage(
        self,
        tier: TierType,
        timestamp_ms: int,
    ) -> float:
        """
        Get LULD band percentage based on tier and time of day.

        Tier 1 (S&P 500, Russell 1000):
        - 9:30-9:45 ET: 10% bands
        - 9:45-15:35 ET: 5% bands
        - 15:35-16:00 ET: 10% bands

        Tier 2 (all others):
        - All day: 10% bands
        """
        if tier == TierType.TIER2:
            return LULD_TIER2_BAND

        # Convert timestamp to Eastern Time
        try:
            dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
            # Simplified: assume UTC-5 for Eastern (doesn't account for DST)
            et_hour = (dt.hour - 5) % 24
            et_minute = dt.minute
            et_time = time(et_hour, et_minute)

            if et_time < NYSE_EARLY_OPEN_END or et_time >= NYSE_LATE_START:
                return LULD_TIER1_OPEN_BAND
            else:
                return LULD_TIER1_REGULAR_BAND
        except Exception:
            return LULD_TIER1_REGULAR_BAND

    def _update_luld_bands(
        self,
        symbol: str,
        reference_price: float,
        timestamp_ms: int,
    ) -> LULDBands:
        """Update LULD bands for a symbol."""
        symbol = symbol.upper()
        tier = self._get_tier(symbol)
        band_pct = self._get_luld_band_percentage(tier, timestamp_ms)

        bands = LULDBands(
            symbol=symbol,
            reference_price=reference_price,
            upper_band=reference_price * (1 + band_pct),
            lower_band=reference_price * (1 - band_pct),
            band_percentage=band_pct,
            tier=tier,
            last_update_ms=timestamp_ms,
        )

        self._luld_bands[symbol] = bands
        return bands

    def get_luld_bands(self, symbol: str) -> Optional[LULDBands]:
        """Get current LULD bands for a symbol."""
        with self._lock:
            return self._luld_bands.get(symbol.upper())

    # =========================
    # Halt Checking
    # =========================

    def check_luld_violation(
        self,
        symbol: str,
        price: float,
        timestamp_ms: int,
    ) -> HaltStatus:
        """
        Check if a price violates LULD bands.

        Args:
            symbol: Trading symbol
            price: Current/proposed price
            timestamp_ms: Current timestamp

        Returns:
            HaltStatus with halt info if triggered
        """
        if not self._config.enabled or not self._config.luld_enabled:
            return HaltStatus(symbol=symbol)

        with self._lock:
            symbol = symbol.upper()

            # Get or create LULD bands
            bands = self._luld_bands.get(symbol)
            if bands is None:
                ref_price = self._reference_prices.get(
                    symbol, self._prev_close_prices.get(symbol, price)
                )
                bands = self._update_luld_bands(symbol, ref_price, timestamp_ms)

            # Check existing halt
            existing = self._symbol_halts.get(symbol)
            if existing and existing.is_halted:
                if existing.halt_end_ms and timestamp_ms >= existing.halt_end_ms:
                    # Halt expired
                    existing.is_halted = False
                    existing.halt_type = HaltType.NONE
                else:
                    return existing

            # Check band violation
            if price > bands.upper_band:
                halt = HaltStatus(
                    symbol=symbol,
                    halt_type=HaltType.LULD_PAUSE,
                    is_halted=True,
                    halt_start_ms=timestamp_ms,
                    halt_end_ms=timestamp_ms + (self._config.luld_pause_duration_sec * 1000),
                    reason=f"Limit Up: price ${price:.2f} > upper band ${bands.upper_band:.2f}",
                    trigger_price=price,
                    resume_price=bands.upper_band,
                )
                self._symbol_halts[symbol] = halt
                logger.warning(f"LULD Limit Up triggered for {symbol}: {halt.reason}")
                return halt

            elif price < bands.lower_band:
                halt = HaltStatus(
                    symbol=symbol,
                    halt_type=HaltType.LULD_PAUSE,
                    is_halted=True,
                    halt_start_ms=timestamp_ms,
                    halt_end_ms=timestamp_ms + (self._config.luld_pause_duration_sec * 1000),
                    reason=f"Limit Down: price ${price:.2f} < lower band ${bands.lower_band:.2f}",
                    trigger_price=price,
                    resume_price=bands.lower_band,
                )
                self._symbol_halts[symbol] = halt
                logger.warning(f"LULD Limit Down triggered for {symbol}: {halt.reason}")
                return halt

            return HaltStatus(symbol=symbol)

    def check_market_wide_halt(
        self,
        spx_return: float,
        timestamp_ms: int,
    ) -> HaltStatus:
        """
        Check for market-wide circuit breaker.

        Args:
            spx_return: S&P 500 return from previous close (negative for decline)
            timestamp_ms: Current timestamp

        Returns:
            HaltStatus for market-wide halt
        """
        if not self._config.enabled or not self._config.mwcb_enabled:
            return HaltStatus(symbol="MARKET")

        with self._lock:
            # Check existing halt
            if self._market_halt.is_halted:
                if self._market_halt.halt_end_ms and timestamp_ms >= self._market_halt.halt_end_ms:
                    # Halt expired
                    self._market_halt.is_halted = False
                    self._market_halt.halt_type = HaltType.NONE
                else:
                    return self._market_halt

            # Check time of day (MWCB only triggers before 3:25pm ET)
            try:
                dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                et_hour = (dt.hour - 5) % 24
                et_minute = dt.minute
                et_time = time(et_hour, et_minute)

                # After 3:25pm, only Level 3 triggers
                after_cutoff = et_time >= NYSE_MWCB_CUTOFF
            except Exception:
                after_cutoff = False

            # Check circuit breaker levels
            if spx_return <= MWCB_LEVEL3_THRESHOLD:
                if "LEVEL3" not in self._mwcb_triggered_today:
                    self._mwcb_triggered_today.add("LEVEL3")
                    halt = HaltStatus(
                        symbol="MARKET",
                        halt_type=HaltType.MWCB_LEVEL3,
                        is_halted=True,
                        halt_start_ms=timestamp_ms,
                        halt_end_ms=None,  # Remainder of day
                        reason=f"Market-Wide Circuit Breaker Level 3: S&P 500 down {spx_return:.1%}",
                        trigger_price=spx_return,
                    )
                    self._market_halt = halt
                    logger.critical(f"MWCB Level 3 triggered: {halt.reason}")
                    return halt

            elif spx_return <= MWCB_LEVEL2_THRESHOLD and not after_cutoff:
                if "LEVEL2" not in self._mwcb_triggered_today:
                    self._mwcb_triggered_today.add("LEVEL2")
                    halt = HaltStatus(
                        symbol="MARKET",
                        halt_type=HaltType.MWCB_LEVEL2,
                        is_halted=True,
                        halt_start_ms=timestamp_ms,
                        halt_end_ms=timestamp_ms + (self._config.mwcb_halt_duration_sec * 1000),
                        reason=f"Market-Wide Circuit Breaker Level 2: S&P 500 down {spx_return:.1%}",
                        trigger_price=spx_return,
                    )
                    self._market_halt = halt
                    logger.critical(f"MWCB Level 2 triggered: {halt.reason}")
                    return halt

            elif spx_return <= MWCB_LEVEL1_THRESHOLD and not after_cutoff:
                if "LEVEL1" not in self._mwcb_triggered_today:
                    self._mwcb_triggered_today.add("LEVEL1")
                    halt = HaltStatus(
                        symbol="MARKET",
                        halt_type=HaltType.MWCB_LEVEL1,
                        is_halted=True,
                        halt_start_ms=timestamp_ms,
                        halt_end_ms=timestamp_ms + (self._config.mwcb_halt_duration_sec * 1000),
                        reason=f"Market-Wide Circuit Breaker Level 1: S&P 500 down {spx_return:.1%}",
                        trigger_price=spx_return,
                    )
                    self._market_halt = halt
                    logger.warning(f"MWCB Level 1 triggered: {halt.reason}")
                    return halt

            return HaltStatus(symbol="MARKET")

    def check_halt_status(
        self,
        symbol: str,
        price: float,
        timestamp_ms: int,
        spx_return: Optional[float] = None,
    ) -> HaltStatus:
        """
        Check complete halt status for a trade.

        Args:
            symbol: Trading symbol
            price: Current/proposed price
            timestamp_ms: Current timestamp
            spx_return: S&P 500 return (for market-wide check)

        Returns:
            HaltStatus (most severe halt if multiple apply)
        """
        if not self._config.enabled:
            return HaltStatus(symbol=symbol)

        # Check market-wide halt first
        if spx_return is not None:
            market_status = self.check_market_wide_halt(spx_return, timestamp_ms)
            if market_status.is_halted:
                return market_status
        elif self._market_halt.is_halted:
            if (
                self._market_halt.halt_end_ms is None
                or timestamp_ms < self._market_halt.halt_end_ms
            ):
                return self._market_halt

        # Check symbol-level LULD
        return self.check_luld_violation(symbol, price, timestamp_ms)

    def is_trading_allowed(
        self,
        symbol: str,
        price: float,
        timestamp_ms: int,
    ) -> Tuple[bool, Optional[HaltStatus]]:
        """
        Check if trading is allowed for a symbol.

        Args:
            symbol: Trading symbol
            price: Current/proposed price
            timestamp_ms: Current timestamp

        Returns:
            (is_allowed, halt_status) tuple
        """
        status = self.check_halt_status(symbol, price, timestamp_ms)
        return (not status.is_halted, status if status.is_halted else None)

    # =========================
    # News Halts (Simulated)
    # =========================

    def trigger_news_halt(
        self,
        symbol: str,
        timestamp_ms: int,
        duration_sec: int = 30 * 60,
        reason: str = "News pending",
    ) -> HaltStatus:
        """
        Trigger a news halt for a symbol.

        Args:
            symbol: Symbol to halt
            timestamp_ms: Current timestamp
            duration_sec: Halt duration in seconds
            reason: Halt reason

        Returns:
            HaltStatus for the halt
        """
        with self._lock:
            symbol = symbol.upper()
            halt = HaltStatus(
                symbol=symbol,
                halt_type=HaltType.NEWS_PENDING,
                is_halted=True,
                halt_start_ms=timestamp_ms,
                halt_end_ms=timestamp_ms + (duration_sec * 1000),
                reason=reason,
            )
            self._symbol_halts[symbol] = halt
            logger.info(f"News halt triggered for {symbol}: {reason}")
            return halt

    def clear_halt(self, symbol: str) -> None:
        """Clear halt for a symbol."""
        with self._lock:
            symbol = symbol.upper()
            if symbol in self._symbol_halts:
                self._symbol_halts[symbol].is_halted = False
                self._symbol_halts[symbol].halt_type = HaltType.NONE

    # =========================
    # State Export
    # =========================

    def to_dict(self) -> Dict[str, Any]:
        """Export state for persistence."""
        with self._lock:
            return {
                "market_halt": {
                    "halt_type": self._market_halt.halt_type.value,
                    "is_halted": self._market_halt.is_halted,
                    "halt_start_ms": self._market_halt.halt_start_ms,
                    "halt_end_ms": self._market_halt.halt_end_ms,
                    "reason": self._market_halt.reason,
                },
                "symbol_halts": {
                    sym: {
                        "halt_type": h.halt_type.value,
                        "is_halted": h.is_halted,
                        "halt_end_ms": h.halt_end_ms,
                    }
                    for sym, h in self._symbol_halts.items()
                    if h.is_halted
                },
                "luld_bands": {
                    sym: {
                        "reference_price": b.reference_price,
                        "upper_band": b.upper_band,
                        "lower_band": b.lower_band,
                        "band_percentage": b.band_percentage,
                    }
                    for sym, b in self._luld_bands.items()
                },
                "mwcb_triggered_today": list(self._mwcb_triggered_today),
            }


# =========================
# Factory Functions
# =========================


def create_trading_halts_simulator(
    asset_class: str = "equity",
    luld_enabled: bool = True,
    mwcb_enabled: bool = True,
) -> TradingHaltsSimulator:
    """
    Create a trading halts simulator.

    Args:
        asset_class: "equity" or "crypto" (crypto returns disabled simulator)
        luld_enabled: Enable LULD circuit breakers
        mwcb_enabled: Enable market-wide circuit breakers

    Returns:
        TradingHaltsSimulator instance
    """
    if asset_class != "equity":
        # Return disabled simulator for non-equity
        config = TradingHaltsConfig(enabled=False)
    else:
        config = TradingHaltsConfig(
            luld_enabled=luld_enabled,
            mwcb_enabled=mwcb_enabled,
            enabled=True,
        )
    return TradingHaltsSimulator(config)


def create_disabled_halts_simulator() -> TradingHaltsSimulator:
    """Create a disabled trading halts simulator (for crypto or testing)."""
    return TradingHaltsSimulator(TradingHaltsConfig(enabled=False))
